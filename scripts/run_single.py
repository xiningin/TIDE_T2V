from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.core.agent import VistaAgent  # noqa: E402
from vista.utils.config import deep_update, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TIDE training or one-pass deployment.")
    parser.add_argument("--prompt", required=True, help="User text-to-video prompt.")
    parser.add_argument("--mode", choices=("train", "deploy", "rewrite"), default="train", help="train: video-level learning; deploy: one rewrite + one video; rewrite: text-only.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--output-dir", default=None, help="Output directory for videos and trajectory.")
    parser.add_argument("--mllm-provider", default=None, help="MLLM provider override, for example ark or vllm.")
    parser.add_argument("--mllm-base-url", default=None, help="MLLM base URL override. For vLLM use http://host:port/v1 or http://host:port.")
    parser.add_argument("--api-key", default=None, help="MLLM API key. Empty keeps MLLM dry-run if configured.")
    parser.add_argument("--model", default=None, help="MLLM model name override.")
    parser.add_argument("--model-path", default=None, help="Local MLLM model path override. Useful when vLLM serves a local model path.")
    parser.add_argument("--ckpt-dir", default=None, help="Wan checkpoint directory override.")
    parser.add_argument("--wan-dir", default=None, help="Wan code directory override.")
    parser.add_argument("--python-executable", default=None, help="Wan inference Python executable override.")
    parser.add_argument("--iterations", type=int, default=None, help="Self-improvement iteration override.")
    parser.add_argument("--real-run", action="store_true", help="Disable Wan dry-run. Requires a real ckpt dir.")
    parser.add_argument("--t2v-dry-run", action="store_true", help="Force Wan dry-run.")
    parser.add_argument("--skill-dir", default=None, help="Versioned rewrite skill directory.")
    parser.add_argument("--rules-file", default=None, help="Initialization or promoted rules JSON used by the rewriter.")
    parser.add_argument("--skill-learning", action="store_true", help="Enable training-time experience distillation and candidate patch logging.")
    parser.add_argument("--activation-mode", choices=("normal", "all", "random"), default=None)
    parser.add_argument("--random-activation-count", type=int, default=None)
    parser.add_argument("--random-activation-seed", type=int, default=None)
    parser.add_argument("--freeze-negative-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides: dict = {"mllm": {}, "t2v": {}, "run": {}, "skill_learning": {}}
    overrides["run"]["mode"] = args.mode
    if args.skill_dir:
        overrides["skill_learning"]["skill_dir"] = args.skill_dir
    if args.rules_file:
        overrides["skill_learning"]["rules_file"] = args.rules_file
    if args.skill_learning:
        overrides["skill_learning"]["enabled"] = True
    if args.activation_mode is not None:
        overrides["skill_learning"]["activation_mode"] = args.activation_mode
    if args.random_activation_count is not None:
        overrides["skill_learning"]["random_activation_count"] = args.random_activation_count
    if args.random_activation_seed is not None:
        overrides["skill_learning"]["random_activation_seed"] = args.random_activation_seed
    if args.freeze_negative_prompt:
        overrides["skill_learning"]["freeze_negative_prompt"] = True
    if args.mllm_provider:
        overrides["mllm"]["provider"] = args.mllm_provider
    if args.mllm_base_url:
        overrides["mllm"]["base_url"] = args.mllm_base_url
    if args.api_key is not None:
        overrides["mllm"]["api_key"] = args.api_key
    if args.model:
        overrides["mllm"]["model"] = args.model
    if args.model_path:
        overrides["mllm"]["model_path"] = args.model_path
        overrides["mllm"]["model"] = args.model_path
    if args.ckpt_dir:
        overrides["t2v"]["ckpt_dir"] = args.ckpt_dir
    if args.wan_dir:
        overrides["t2v"]["wan_dir"] = args.wan_dir
    if args.python_executable:
        overrides["t2v"]["python_executable"] = args.python_executable
    if args.iterations is not None:
        overrides["run"]["iterations"] = args.iterations
    if args.real_run:
        overrides["t2v"]["dry_run"] = False
    if args.t2v_dry_run:
        overrides["t2v"]["dry_run"] = True
    config = deep_update(config, {key: value for key, value in overrides.items() if value})

    agent = VistaAgent(config)
    if args.mode == "rewrite":
        result = agent.rewrite_prompt(args.prompt)
    elif args.mode == "deploy":
        result = agent.run_deploy(args.prompt, output_dir=args.output_dir)
    else:
        result = agent.run(args.prompt, output_dir=args.output_dir)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
