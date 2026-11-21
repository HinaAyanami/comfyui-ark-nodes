"""火山方舟（Volcengine Ark）Seedream 出图节点 for ComfyUI.

- 走官方 images/generations REST 接口，不依赖任何 SDK
- API Key 读取顺序：环境变量 ARK_API_KEY -> 本目录 api_key.txt -> Hermes 的 .env
- 默认模型对齐 seedream 5.0（flash 便宜 / pro 精修）
"""

import base64
import io
import json
import os
import random
import time

import numpy as np
import requests
import torch
from PIL import Image

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ENDPOINT = BASE_URL + "/images/generations"

MODELS = [
    "doubao-seedream-5-0-flash-260915",
    "doubao-seedream-5-0-pro-260628",
]

# 2K 组在前（长边 2048 起），全部压在 2.61MP 以内
# —— pro 超过 2.61MP 会按 2 张计费
SIZES = {
    "2K 16:9  2048x1152 (2.36MP)": "2048x1152",
    "2K 9:16  1152x2048 (2.36MP)": "1152x2048",
    "2K 1:1   1600x1600 (2.56MP)": "1600x1600",
    "2K 4:3   1856x1392 (2.58MP)": "1856x1392",
    "2K 3:4   1392x1856 (2.58MP)": "1392x1856",
    "2K 3:2   1920x1280 (2.46MP)": "1920x1280",
    "2K 2:3   1280x1920 (2.46MP)": "1280x1920",
    "2K 21:9  2304x987  (2.27MP)": "2304x987",
    "1:1   1536x1536 (2.36MP)": "1536x1536",
    "16:9  1920x1088 (2.09MP)": "1920x1088",
    "9:16  1088x1920 (2.09MP)": "1088x1920",
    "4:3   1664x1248 (2.08MP)": "1664x1248",
    "3:4   1248x1664 (2.08MP)": "1248x1664",
    "3:2   1776x1184 (2.10MP)": "1776x1184",
    "2:3   1184x1776 (2.10MP)": "1184x1776",
    "21:9  2048x880  (1.80MP)": "2048x880",
}

MP_LIMIT = 2.61


def _load_key() -> str:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        return key

    local = os.path.join(os.path.dirname(__file__), "api_key.txt")
    if os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if txt:
            return txt.strip().strip('"').strip("'")

    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", ".env"),
        os.path.join(os.path.expanduser("~"), ".env"),
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ARK_API_KEY"):
                        _, _, val = line.partition("=")
                        val = val.strip().strip('"').strip("'")
                        if val:
                            return val
        except OSError:
            pass
    return ""


def _tensor_to_data_uri(img: torch.Tensor) -> str:
    arr = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bytes_to_tensor(raw: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


class ArkSeedreamImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (MODELS, {"default": MODELS[0]}),
                "size_preset": (list(SIZES.keys()), {"default": list(SIZES.keys())[0]}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1,
                                 "control_after_generate": True}),
                "timeout": ("INT", {"default": 180, "min": 30, "max": 600, "step": 10}),
            },
            "optional": {
                "custom_size": ("STRING", {"default": "", "multiline": False}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = "火山方舟/Seedream"

    def run(self, prompt, model, size_preset, max_images, watermark, seed,
            timeout, custom_size="", image1=None, image2=None, image3=None,
            image4=None, api_key_override=""):
        key = (api_key_override or "").strip() or _load_key()
        if not key:
            raise RuntimeError("没找到火山方舟 API Key：设环境变量 ARK_API_KEY，"
                               "或在本节点目录放 api_key.txt")

        size = (custom_size or "").strip() or SIZES[size_preset]
        try:
            _w, _h = (int(v) for v in size.lower().split("x"))
            _mp = _w * _h / 1e6
            mp_note = f"{_mp:.2f}MP"
            if _mp > MP_LIMIT:
                mp_note += " ⚠超过2.61MP，pro 按 2 张计费"
        except Exception:
            mp_note = size
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 180
        try:
            n_images = max(1, int(max_images))
        except (TypeError, ValueError):
            n_images = 1

        refs = [x for x in (image1, image2, image3, image4) if x is not None]
        ref_uris = [_tensor_to_data_uri(t) for t in refs] if refs else []

        t0 = time.time()
        tensors = []
        req_ids = []
        for i in range(n_images):
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "response_format": "url",
                "watermark": bool(watermark),
            }
            # seedream 5.x 不支持 sequential_image_generation 一次出组图，
            # 这里改成多次独立请求；4.x 仍走原生组图参数
            if "seedream-4" in model:
                if n_images > 1:
                    payload["sequential_image_generation"] = "auto"
                    payload["max_images"] = n_images
                if seed is not None and seed >= 0:
                    payload["seed"] = int(seed)
            elif seed is not None and seed >= 0:
                payload["seed"] = int(seed) + i
            if ref_uris:
                payload["image"] = ref_uris[0] if len(ref_uris) == 1 else ref_uris

            resp = requests.post(
                ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
                json=payload,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"方舟接口报错 {resp.status_code}: {resp.text[:800]}")
            req_ids.append(resp.headers.get("X-Request-Id", "-"))

            data = resp.json().get("data") or []
            if not data:
                raise RuntimeError("接口没返回图片：" + resp.text[:800])
            for j, item in enumerate(data):
                if item.get("url"):
                    got = requests.get(item["url"], timeout=120)
                    got.raise_for_status()
                    tensors.append(_bytes_to_tensor(got.content))
                elif item.get("b64_json"):
                    tensors.append(_bytes_to_tensor(
                        base64.b64decode(item["b64_json"])))
                else:
                    raise RuntimeError(
                        f"第 {j} 张既没有 url 也没有 b64：" + json.dumps(item)[:300])
            if "seedream-4" in model:
                break  # 4.x 一次请求就把组图拿全了

        batch = torch.cat(tensors, dim=0)
        info = (f"model={model}  size={size} ({mp_note})  images={len(tensors)}  "
                f"refs={len(refs)}  {time.time() - t0:.1f}s  "
                f"request_id={req_ids[0] if req_ids else '-'}")
        print("[火山方舟] " + info)
        return (batch, info)


NODE_CLASS_MAPPINGS = {"ArkSeedreamImage": ArkSeedreamImage}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkSeedreamImage": "火山方舟 Seedream 出图"}
