"""图片处理工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps


def normalize_image(path: str | Path, max_size: int = 1024) -> Image.Image:
    """读图并规范化：转 RGB，按比例缩放到 max_size 以内。"""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋转
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        if w >= h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def get_image_info(path: str | Path) -> Tuple[int, int, int] | None:
    """快速获取 (width, height, file_size)，不读完整图像。"""
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w, h, Path(path).stat().st_size
    except Exception:
        return None


def is_animated(path: str | Path) -> bool:
    """判断是否是动图。"""
    try:
        with Image.open(path) as img:
            return bool(getattr(img, "is_animated", False))
    except Exception:
        return False
