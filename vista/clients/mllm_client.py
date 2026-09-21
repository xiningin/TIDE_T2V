from __future__ import annotations

import json
import base64
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from vista.utils.video_io import extract_video_frames, image_to_data_url


@dataclass
class MLLMResponse:
    text: str
    dry_run: bool = False
    raw: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    elapsed_seconds: float = 0.0
    base_url: str | None = None


class MLLMClient:
    """MLLM client with frame-based video inputs.

    Ark Seed uses the OpenAI SDK Responses API. The legacy DeepSeek-compatible
    chat-completions path is kept for config compatibility.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "ark")
        self.model = config.get("model") or config.get("model_path") or "doubao-seed-2-0-mini-260215"
        self.base_url = config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        self.base_urls = self._normalize_base_urls(config.get("base_urls", self.base_url))
        if self.base_urls:
            self.base_url = self.base_urls[0]
        self.batch_size = max(1, int(config.get("batch_size", 1)))
        self._base_url_lock = threading.Lock()
        self._base_url_index = 0
        self.api_key_required = bool(config.get("api_key_required", self.provider != "vllm"))
        self.api_key = config.get("api_key") or os.getenv(config.get("api_key_env", "ARK_API_KEY"), "")
        if self.provider == "vllm" and not self.api_key:
            self.api_key = "EMPTY"
        self.temperature = float(config.get("temperature", 0.2))
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.timeout = int(config.get("timeout_seconds", 120))
        self.max_retries = int(config.get("max_retries", 2))
        self.retry_backoff_seconds = float(config.get("retry_backoff_seconds", 1.0))
        self.transient_retry_backoff_seconds = float(config.get("transient_retry_backoff_seconds", 60.0))
        self.max_retry_backoff_seconds = float(config.get("max_retry_backoff_seconds", 300.0))
        self.video_input = config.get("video_input", {})
        self.dry_run = bool(
            config.get("dry_run", False)
            or (config.get("dry_run_when_api_key_missing", True) and self.api_key_required and not self.api_key)
        )
        self._local_model: Any | None = None
        self._local_processor: Any | None = None
        self._local_torch: Any | None = None
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        prompt: str,
        *,
        videos: list[str] | None = None,
        expected: str = "text",
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> MLLMResponse:
        started = perf_counter()
        if self.dry_run:
            response = MLLMResponse(text=self._dry_run_response(prompt, expected), dry_run=True, usage=self._zero_usage())
        elif self.provider == "ark":
            response = self._call_ark_responses(prompt, videos=videos, temperature=temperature)
        elif self.provider == "vllm":
            response = self._call_vllm_chat(
                prompt,
                videos=videos,
                temperature=temperature,
                thinking=thinking,
            )
        elif self.provider == "transformers":
            response = self._call_transformers_local(prompt, videos=videos, temperature=temperature)
        else:
            response = self._call_openai_compatible_chat(
                prompt,
                videos=videos,
                temperature=temperature,
                thinking=thinking,
            )

        response.elapsed_seconds = perf_counter() - started
        if response.usage is None:
            response.usage = self._extract_usage(response.raw)
        self.calls.append(
            {
                "provider": self.provider,
                "model": self.model,
                "expected": expected,
                "video_count": len(videos or []),
                "dry_run": response.dry_run,
                "elapsed_seconds": response.elapsed_seconds,
                "base_url": response.base_url or self.base_url,
                "batch_size": self.batch_size,
                "thinking": thinking,
                "usage": response.usage,
            }
        )
        return response

    def _call_ark_responses(
        self,
        prompt: str,
        *,
        videos: list[str] | None,
        temperature: float | None,
    ) -> MLLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            return self._call_ark_responses_http(prompt, videos=videos, temperature=temperature)

        content: list[dict[str, Any]] = []
        if videos:
            for video_path in videos:
                frames = extract_video_frames(
                    video_path,
                    max_frames=int(self.video_input.get("max_frames", 4)),
                    jpeg_quality=int(self.video_input.get("jpeg_quality", 85)),
                )
                for frame in frames:
                    content.append({"type": "input_image", "image_url": image_to_data_url(frame)})
        content.append({"type": "input_text", "text": prompt})

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    temperature=self.temperature if temperature is None else temperature,
                    max_output_tokens=self.max_tokens,
                )
                text = getattr(response, "output_text", None) or self._extract_responses_text(response)
                raw = response.model_dump() if hasattr(response, "model_dump") else None
                return MLLMResponse(text=text, raw=raw, usage=self._extract_usage(raw))
            except Exception as exc:  # noqa: BLE001 - preserve API error details in final failure.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay_seconds(attempt, exc))
        raise RuntimeError(f"Ark MLLM request failed after retries: {last_error}") from last_error

    def _call_ark_responses_http(
        self,
        prompt: str,
        *,
        videos: list[str] | None,
        temperature: float | None,
    ) -> MLLMResponse:
        content: list[dict[str, Any]] = []
        if videos:
            for video_path in videos:
                frames = extract_video_frames(
                    video_path,
                    max_frames=int(self.video_input.get("max_frames", 4)),
                    jpeg_quality=int(self.video_input.get("jpeg_quality", 85)),
                )
                for frame in frames:
                    content.append({"type": "input_image", "image_url": image_to_data_url(frame)})
        content.append({"type": "input_text", "text": prompt})

        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url.rstrip('/')}/responses"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                text = data.get("output_text") or self._extract_responses_text_from_dict(data)
                return MLLMResponse(text=text, raw=data, usage=self._extract_usage(data))
            except Exception as exc:  # noqa: BLE001 - preserve API error details in final failure.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay_seconds(attempt, exc))
        raise RuntimeError(f"Ark MLLM HTTP request failed after retries: {last_error}") from last_error

    def _call_openai_compatible_chat(
        self,
        prompt: str,
        *,
        videos: list[str] | None,
        temperature: float | None,
        thinking: bool | None = None,
    ) -> MLLMResponse:
        content: list[dict[str, Any]] = []
        frame_groups: list[list[dict[str, Any]]] = []
        video_mode = str(self.video_input.get("mode", "frames"))
        if videos and video_mode == "base64_video":
            fps = float(self.video_input.get("fps", 2))
            for video_path in videos:
                encoded = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
                        "fps": int(fps) if fps.is_integer() else fps,
                    }
                )
        elif videos:
            for video_path in videos:
                frames = extract_video_frames(
                    video_path,
                    max_frames=int(self.video_input.get("max_frames", 4)),
                    jpeg_quality=int(self.video_input.get("jpeg_quality", 85)),
                )
                group: list[dict[str, Any]] = []
                for frame in frames:
                    group.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame)}})
                frame_groups.append(group)
                content.extend(group)
        prompt_content = {"type": "text", "text": prompt}
        content.append(prompt_content)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                extra_body = self.config.get("extra_body")
                if isinstance(extra_body, dict):
                    payload.update(extra_body)
                if thinking is not None:
                    payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
                response = requests.post(self._chat_completions_url(self.base_url), headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                text = message.get("content")
                reasoning = message.get("reasoning_content") or ""
                usage = self._extract_usage(data)
                if not text and reasoning:
                    # DeepSeek reasoning endpoints can spend the complete output
                    # budget on reasoning and return content=null. Preserve the
                    # requested long-thinking pass, then use a bounded non-thinking
                    # pass only to serialize its conclusion into the requested form.
                    final_content = list(content)
                    final_content.append(
                        {
                            "type": "text",
                            "text": (
                                "The previous deliberation reached the token limit before "
                                "the final answer. Based only on the analysis below, return "
                                "the final answer in the exact format required by the original "
                                "instruction. Do not add reasoning or markdown.\n\n"
                                f"Previous analysis:\n{reasoning[-24000:]}"
                            ),
                        }
                    )
                    final_payload = {
                        **payload,
                        "messages": [{"role": "user", "content": final_content}],
                        "temperature": 0,
                        "max_tokens": min(4096, self.max_tokens),
                        "thinking": {"type": "disabled"},
                    }
                    final_response = requests.post(
                        self._chat_completions_url(self.base_url),
                        headers=headers,
                        json=final_payload,
                        timeout=self.timeout,
                    )
                    final_response.raise_for_status()
                    final_data = final_response.json()
                    text = final_data["choices"][0]["message"].get("content")
                    if not text:
                        raise RuntimeError("Reasoning finalizer returned content=null")
                    usage = self._sum_usage_dicts(usage, self._extract_usage(final_data))
                    final_data["psista_reasoning_finalizer"] = {
                        "used": True,
                        "reasoning_chars": len(reasoning),
                    }
                    data = final_data
                if not text:
                    raise RuntimeError("MLLM returned content=null without usable reasoning")
                text = self._maybe_strip_thinking_text(text)
                return MLLMResponse(text=text, raw=data, usage=usage)
            except Exception as exc:  # noqa: BLE001 - preserve API error details in final failure.
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    # DashScope returns the actionable request ID and validation
                    # message in the response body. Keep it bounded so a failed
                    # multimedia request cannot flood the experiment log.
                    detail = exc.response.text.replace("\n", " ").strip()[:2000]
                    last_error = RuntimeError(f"{exc}; response_body={detail}")
                else:
                    last_error = exc
                if attempt < self.max_retries:
                    if frame_groups and self._is_data_inspection_error(exc):
                        payload["messages"][0]["content"] = self._inspection_retry_content(
                            frame_groups,
                            prompt_content,
                            retry_number=attempt + 1,
                        )
                    time.sleep(self._retry_delay_seconds(attempt, exc))
        raise RuntimeError(f"MLLM request failed after retries: {last_error}") from last_error

    def _call_vllm_chat(
        self,
        prompt: str,
        *,
        videos: list[str] | None,
        temperature: float | None,
        thinking: bool | None,
    ) -> MLLMResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if videos:
            for video_path in videos:
                frames = extract_video_frames(
                    video_path,
                    max_frames=int(self.video_input.get("max_frames", 4)),
                    jpeg_quality=int(self.video_input.get("jpeg_quality", 85)),
                )
                for frame in frames:
                    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame)}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        chat_template_kwargs = self._chat_template_kwargs()
        if thinking is not None:
            chat_template_kwargs["enable_thinking"] = thinking
            chat_template_kwargs["preserve_thinking"] = False
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        extra_body = self.config.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        last_base_url: str | None = None
        for attempt in range(self.max_retries + 1):
            base_url = self._next_base_url()
            last_base_url = base_url
            try:
                response = requests.post(self._chat_completions_url(base_url), headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                text = message.get("content")
                reasoning = message.get("reasoning_content") or ""
                if not text:
                    # A reasoning model can consume its whole generation budget before
                    # emitting the final answer. Some vLLM/Qwen responses also omit
                    # reasoning_content entirely when this happens. In both cases ask a
                    # bounded non-thinking pass to produce the required final structure
                    # instead of repeating the expensive long-thinking request.
                    final_content = list(content)
                    if reasoning:
                        finalizer_instruction = (
                            "The previous deliberation reached the token limit before "
                            "a final answer. Using the analysis below, now return only "
                            "the final answer in the exact format requested by the "
                            "original instruction. Do not add new reasoning.\n\n"
                            f"Previous analysis:\n{reasoning[-24000:]}"
                        )
                    else:
                        finalizer_instruction = (
                            "The previous long-thinking pass returned no serializable "
                            "answer. Inspect the original inputs and return only the final "
                            "answer in the exact format requested by the original "
                            "instruction. Do not add reasoning or markdown."
                        )
                    final_content.append(
                        {
                            "type": "text",
                            "text": finalizer_instruction,
                        }
                    )
                    final_payload = {
                        **payload,
                        "messages": [{"role": "user", "content": final_content}],
                        "max_tokens": min(2048, self.max_tokens),
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                            "preserve_thinking": False,
                        },
                    }
                    final_response = requests.post(
                        self._chat_completions_url(base_url),
                        headers=headers,
                        json=final_payload,
                        timeout=self.timeout,
                    )
                    final_response.raise_for_status()
                    final_data = final_response.json()
                    text = final_data["choices"][0]["message"].get("content")
                    if not text:
                        raise RuntimeError("Qwen reasoning finalizer returned content=null")
                    usage = self._sum_usage_dicts(
                        self._extract_usage(data),
                        self._extract_usage(final_data),
                    )
                    final_data["psista_reasoning_finalizer"] = {
                        "used": True,
                        "reasoning_chars": len(reasoning),
                    }
                    data = final_data
                else:
                    usage = self._extract_usage(data)
                if not text:
                    raise RuntimeError("Qwen returned content=null without usable reasoning")
                text = self._maybe_strip_thinking_text(text)
                return MLLMResponse(text=text, raw=data, usage=usage, base_url=base_url)
            except Exception as exc:  # noqa: BLE001 - preserve API error details in final failure.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay_seconds(attempt, exc))
        raise RuntimeError(f"vLLM MLLM request failed after retries via {last_base_url}: {last_error}") from last_error

    def _retry_delay_seconds(self, attempt: int, exc: Exception) -> float:
        """Return a bounded retry delay, respecting provider recovery windows."""
        base = self.retry_backoff_seconds
        retry_after = 0.0
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            if status in {408, 409, 425, 429, 500, 502, 503, 504}:
                base = self.transient_retry_backoff_seconds
            header = exc.response.headers.get("Retry-After")
            if header:
                try:
                    retry_after = max(0.0, float(header))
                except ValueError:
                    pass
        delay = max(retry_after, base * (2**attempt))
        return min(self.max_retry_backoff_seconds, delay)

    @staticmethod
    def _is_data_inspection_error(exc: Exception) -> bool:
        if not isinstance(exc, requests.HTTPError) or exc.response is None:
            return False
        if exc.response.status_code != 400:
            return False
        return "datainspectionfailed" in exc.response.text.replace("_", "").lower()

    @staticmethod
    def _inspection_retry_content(
        frame_groups: list[list[dict[str, Any]]],
        prompt_content: dict[str, Any],
        *,
        retry_number: int,
    ) -> list[dict[str, Any]]:
        """Reduce frames per video after provider-side content inspection rejects a batch."""
        reduced: list[dict[str, Any]] = []
        divisor = 2 ** max(1, retry_number)
        for group in frame_groups:
            if not group:
                continue
            count = max(1, (len(group) + divisor - 1) // divisor)
            if count == 1:
                reduced.append(group[len(group) // 2])
                continue
            indices = [round(index * (len(group) - 1) / (count - 1)) for index in range(count)]
            reduced.extend(group[index] for index in indices)
        reduced.append(prompt_content)
        return reduced

    def _chat_completions_url(self, base_url: str | None = None) -> str:
        base_url = (base_url or self.base_url).rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _normalize_base_urls(value: Any) -> list[str]:
        if isinstance(value, str):
            urls = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple)):
            urls = [str(item).strip() for item in value]
        else:
            urls = []
        return [url for url in urls if url]

    def _next_base_url(self) -> str:
        if not self.base_urls:
            return self.base_url
        if len(self.base_urls) == 1:
            return self.base_urls[0]
        with self._base_url_lock:
            base_url = self.base_urls[self._base_url_index % len(self.base_urls)]
            self._base_url_index += 1
        return base_url

    @staticmethod
    def _sum_usage_dicts(*values: dict[str, int] | None) -> dict[str, int]:
        total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for value in values:
            if not value:
                continue
            for key in total:
                total[key] += int(value.get(key) or 0)
        return total

    def _call_transformers_local(
        self,
        prompt: str,
        *,
        videos: list[str] | None,
        temperature: float | None,
    ) -> MLLMResponse:
        model, processor, torch = self._load_transformers_local()
        images = self._load_video_frame_images(videos)
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.config.get("enable_thinking", False)),
            preserve_thinking=bool(self.config.get("preserve_thinking", False)),
        )

        inputs = processor(text=[chat_text], images=images or None, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs, model.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": (self.temperature if temperature is None else temperature) > 0,
            "temperature": self.temperature if temperature is None else temperature,
        }
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text = self._maybe_strip_thinking_text(text)
        return MLLMResponse(text=text, raw={"provider": "transformers", "model": self.model}, usage=self._zero_usage())

    def _load_transformers_local(self) -> tuple[Any, Any, Any]:
        if self._local_model is not None and self._local_processor is not None and self._local_torch is not None:
            return self._local_model, self._local_processor, self._local_torch

        cuda_visible_devices = self.config.get("cuda_visible_devices")
        # Each local evaluator runs in its own process. Its explicit assignment must
        # override a broad parent mask (for example the controller's 0-5 safety mask),
        # otherwise device_map="auto" may shard one Qwen across every visible GPU and
        # collide with the three SANA workers.
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

        self._preload_cuda_runtime_libraries()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = self.config.get("model_path") or self.model
        dtype_name = str(self.config.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        device_map = self.config.get("device_map", "auto")
        self._local_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._local_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self._local_torch = torch
        return self._local_model, self._local_processor, self._local_torch

    @staticmethod
    def _preload_cuda_runtime_libraries() -> None:
        """Expose pip-packaged CUDA 13 NVRTC libraries to PyTorch JIT."""
        import ctypes

        lib_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cu13" / "lib"
        for name in ("libnvrtc-builtins.so.13.0", "libnvrtc.so.13"):
            path = lib_dir / name
            if path.is_file():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    def _load_video_frame_images(self, videos: list[str] | None) -> list[Any]:
        if not videos:
            return []
        from PIL import Image

        images: list[Any] = []
        for video_path in videos:
            frames = extract_video_frames(
                video_path,
                max_frames=int(self.video_input.get("max_frames", 4)),
                jpeg_quality=int(self.video_input.get("jpeg_quality", 85)),
            )
            for frame in frames:
                with Image.open(frame) as image:
                    images.append(image.convert("RGB"))
        return images

    @staticmethod
    def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if hasattr(inputs, "items"):
            return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return inputs

    def _chat_template_kwargs(self) -> dict[str, bool]:
        kwargs: dict[str, bool] = {}
        if "enable_thinking" in self.config:
            kwargs["enable_thinking"] = bool(self.config["enable_thinking"])
        if "preserve_thinking" in self.config:
            kwargs["preserve_thinking"] = bool(self.config["preserve_thinking"])
        return kwargs

    def _maybe_strip_thinking_text(self, text: str) -> str:
        if bool(self.config.get("strip_thinking_output", True)):
            return self._strip_thinking_text(text)
        return text

    @staticmethod
    def _strip_thinking_text(text: str) -> str:
        # Qwen-family models may emit either XML thinking blocks or a visible
        # prose trace followed by a labelled final section. Keep only the
        # final section so downstream artifacts never persist hidden reasoning.
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        final_markers = tuple(
            re.finditer(
                r"(?im)^\s*(?:#{1,6}\s*)?(?:final\s+(?:answer|response|output(?:\s+construction)?|prompt)|rewritten\s+prompt)\s*:\s*",
                text,
            )
        )
        if final_markers:
            text = text[final_markers[-1].end() :].strip()
        return text

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        output = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in output:
            for part in getattr(item, "content", []) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _extract_responses_text_from_dict(data: dict[str, Any]) -> str:
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for part in item.get("content", []) or []:
                text = part.get("text")
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _zero_usage() -> dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @classmethod
    def _extract_usage(cls, raw: dict[str, Any] | None) -> dict[str, int]:
        if not raw:
            return cls._zero_usage()
        usage = raw.get("usage") or {}
        input_tokens = cls._coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        output_tokens = cls._coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_tokens = cls._coerce_int(usage.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _dry_run_response(self, prompt: str, expected: str) -> str:
        user_prompt = self._extract_field(prompt, "User Prompt") or self._extract_field(prompt, "user request") or "the user request"
        if expected == "planner":
            return json.dumps(
                [
                    f"A concise realistic video of {user_prompt}, with a clear main subject, coherent action, natural lighting, stable camera movement, no captions, and no unnecessary scene cuts.",
                    f"A grounded cinematic shot of {user_prompt}, keeping the requested subject centered, using physically plausible motion, and emphasizing the described environment and mood.",
                    f"A simple single-scene video of {user_prompt}, with consistent subject identity, smooth temporal progression, and engaging composition.",
                ],
                ensure_ascii=False,
            )
        if expected == "pairwise":
            return json.dumps(
                {
                    "Decision": "COMPARABLE",
                    "WeightedScoreA": 0.5,
                    "WeightedScoreB": 0.5,
                    "Rationale": "Dry-run mode cannot inspect real video content.",
                }
            )
        if expected in {"judge_normal", "judge_adversarial"}:
            return json.dumps(
                {
                    "visual_fidelity": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "motions_and_dynamics": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "temporal_consistency": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "camera_focus": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "visual_safety": {"score": 10, "justification": "Dry-run placeholder assessment."},
                    "video_prompt_alignment": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "physical_commonsense": {"score": 8, "justification": "Dry-run placeholder assessment."},
                    "engagement": {"score": 8, "justification": "Dry-run placeholder assessment."},
                }
            )
        if expected == "judge_meta":
            return json.dumps(
                {
                    "analysis": "Dry-run synthesis of normal and adversarial critiques.",
                    "final_judgment": "No real video was inspected. Treat this as a pipeline validation artifact.",
                    "scores": {"overall": 8},
                }
            )
        if expected == "dtpa":
            return '```python\n["Clarify the main subject and action.", "Specify stable camera framing and physically plausible motion.", "Remove any unnecessary overlays or scene transitions."]\n```'
        if expected == "prompt_sampler":
            count_match = re.search(r"Propose\s+(\d+)\s+different improved prompt pairs", prompt, re.IGNORECASE)
            count = int(count_match.group(1)) if count_match else 3
            variants = [
                f"A realistic single-scene video of {user_prompt}, variation {index + 1}, with the main subject clearly visible, smooth natural movement, stable camera framing, coherent lighting, no subtitles, and no unnecessary cuts."
                for index in range(count)
            ]
            return json.dumps([{"positive_prompt": item, "negative_prompt": "trajectory drift, identity switching"} for item in variants], ensure_ascii=False)
        if expected == "rewrite":
            source = self._extract_field(prompt, "Source request") or self._extract_field(prompt, "User Prompt") or "the user request"
            rule_ids = list(dict.fromkeys(re.findall(r'"rule_id"\s*:\s*"([^"]+)"', prompt)))
            return json.dumps(
                {
                    "rewritten_prompt": (
                        f"A realistic single-scene video of {source}, preserving every requested subject and action, "
                        "with a clear beginning, coherent natural motion, stable identity, physically plausible interactions, "
                        "restrained camera movement, and no unrequested text or audio."
                    ),
                    "negative_prompt": "trajectory drift, identity switching",
                    "activated_rule_ids": rule_ids,
                },
                ensure_ascii=False,
            )
        if expected == "skill_patch":
            return json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "reject",
                    "reasoning": "Dry-run mode has no causal visual outcome for skill learning.",
                    "edit": {},
                },
                ensure_ascii=False,
            )
        if expected == "skill_rule_batch_update":
            module_match = re.search(r"### Current rule\s*(\{.*?\})\s*### Same-rule", prompt, re.DOTALL)
            patches_match = re.search(r"### Same-rule candidate patches\s*(\[.*?\])\s*### Contract", prompt, re.DOTALL)
            module = json.loads(module_match.group(1)) if module_match else {}
            patches = json.loads(patches_match.group(1)) if patches_match else []
            return json.dumps(
                {
                    "schema_version": "2.0",
                    "status": "update",
                    "reasoning": "Dry-run preserves the selected rule while validating the rule batch pipeline.",
                    "evidence_episode_ids": [str(item.get("episode_id")) for item in patches],
                    "module": module,
                },
                ensure_ascii=False,
            )
        if expected == "trajectory_rule_assessments":
            episode_match = re.search(r'"episode_id"\s*:\s*"([^"]+)"', prompt)
            activated_section = prompt.split("### Activated rules to assess", 1)[-1].split("### Contract", 1)[0]
            rule_ids = list(dict.fromkeys(re.findall(r'"rule_id"\s*:\s*"([^"]+)"', activated_section)))
            return json.dumps(
                {
                    "schema_version": "3.0",
                    "episode_id": episode_match.group(1) if episode_match else "dry-run",
                    "assessments": [
                        {
                            "rule_id": rule_id,
                            "status": "no_change",
                            "reasoning": "Dry-run has no visual evidence for a reusable rule change.",
                            "confidence": 0.0,
                            "edit": {},
                        }
                        for rule_id in rule_ids
                    ],
                },
                ensure_ascii=False,
            )
        if expected == "trajectory_patch":
            match = re.search(r'"episode_id"\s*:\s*"([^"]+)"', prompt)
            return json.dumps(
                {
                    "schema_version": "2.0",
                    "status": "candidate",
                    "episode_id": match.group(1) if match else "dry-run",
                    "reasoning": "Dry-run trajectory compression.",
                    "edit": {
                        "operation": "replace",
                        "rule_id": "vista.motion_physics",
                        "trigger": "The source contains a visible action, interaction, or moving subject.",
                        "instruction": "Translate the requested action into an observable, physically connected timeline with a feasible end state.",
                        "negative_terms": ["sliding motion", "trajectory drift"],
                        "expected_side_effects": ["Monitor over-specified choreography."],
                    },
                },
                ensure_ascii=False,
            )
        return "Dry-run feedback: no real MLLM call was made."

    @staticmethod
    def _extract_field(prompt: str, label: str) -> str | None:
        pattern = re.compile(rf"(?:#+\s*)?(?:-\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*(.*)", re.IGNORECASE)
        lines = prompt.splitlines()
        normalized_label = label.strip().lower()
        for index, line in enumerate(lines):
            match = pattern.search(line)
            if match and match.group(1).strip():
                return match.group(1).strip()
            heading = re.sub(r"^#+\s*", "", line).strip().strip("*: ").lower()
            if heading == normalized_label:
                for following in lines[index + 1 :]:
                    if following.strip():
                        return following.strip()
        marker = f"{label}:"
        if marker not in prompt:
            return None
        after = prompt.split(marker, 1)[1].strip()
        line = after.splitlines()[0].strip()
        return line or None


DeepSeekMLLMClient = MLLMClient
