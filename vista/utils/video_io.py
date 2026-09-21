from __future__ import annotations

import base64
import tempfile
from pathlib import Path


def extract_video_frames(video_path: str | Path, max_frames: int = 4, jpeg_quality: int = 85) -> list[Path]:
    """Extract evenly spaced JPEG frames. Returns an empty list if OpenCV cannot decode the file."""
    try:
        import cv2
    except ImportError:
        return []

    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    cap = cv2.VideoCapture(str(path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        indices = sorted({int(i * max(frame_count - 1, 0) / max(max_frames - 1, 1)) for i in range(max_frames)})
        out_dir = Path(tempfile.mkdtemp(prefix="vista_frames_"))
        frames: list[Path] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = out_dir / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            frames.append(out_path)
        return frames
    finally:
        cap.release()


def image_to_data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
