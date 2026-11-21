"""火山方舟（Volcengine Ark）Seedance 视频生成节点 for ComfyUI.

接口：POST /api/v3/contents/generations/tasks（异步），再轮询 GET .../tasks/{id}。
输出 ComfyUI 原生 VIDEO 类型，可直接接 SaveVideo 存盘。

API Key 读取顺序与环境变量同上（ARK_API_KEY -> api_key.txt -> Hermes .env）。
"""

import base64
import io
import json
import os
import time

import requests
import torch
import numpy as np
from PIL import Image

import folder_paths
from comfy_api.latest import InputImpl

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
TASKS_URL = BASE_URL + "/contents/generations/tasks"

# 1.0 系列在这台机器上未开通（返回 NotFound），默认只列 2.x
MODELS = [
    "doubao-seedance-2-5-260628",
    "doubao-seedance-2-0-260128",
]

MODES = ["文生视频", "首帧图生视频", "首尾帧图生视频", "参考图生视频"]
RESOLUTIONS = ["480p", "720p", "1080p", "4k"]
RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]


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

    for path in [os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", ".env"),
                 os.path.join(os.path.expanduser("~"), ".env")]:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("ARK_API_KEY"):
                        val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                        if val:
                            return val
        except OSError:
            pass
    return ""


def _tensor_to_data_uri(img: torch.Tensor) -> str:
    arr = (img.cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    if arr.ndim == 4:
        arr = arr[0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _last_frame_tensor(raw: bytes) -> torch.Tensor:
    """把视频最后一帧解成 IMAGE，用于接到下一段的首帧。"""
    import av

    last = None
    with av.open(io.BytesIO(raw), mode="r") as container:
        for frame in container.decode(container.streams.video[0]):
            last = frame
    if last is None:
        raise RuntimeError("视频里没有可解码的帧，拿不到尾帧")
    pil = last.to_image().convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


class ArkSeedanceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (MODELS, {"default": MODELS[0]}),
                "mode": (MODES, {"default": "文生视频"}),
                "resolution": (RESOLUTIONS, {"default": "720p"}),
                "ratio": (RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "output_format": (["mp4", "mov"], {"default": "mp4"}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 3600, "step": 30}),
            },
            "optional": {
                "image_first": ("IMAGE",),
                "image_last": ("IMAGE",),
                "ref_image1": ("IMAGE",),
                "ref_image2": ("IMAGE",),
                "ref_image3": ("IMAGE",),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "api_key_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "info", "last_frame")
    FUNCTION = "run"
    CATEGORY = "火山方舟/Seedance"

    def run(self, prompt, model, mode, resolution, ratio, duration, generate_audio,
            watermark, output_format, timeout, image_first=None, image_last=None,
            ref_image1=None, ref_image2=None, ref_image3=None,
            custom_model="", api_key_override=""):
        key = (api_key_override or "").strip() or _load_key()
        if not key:
            raise RuntimeError("没找到火山方舟 API Key：设 ARK_API_KEY 或放 api_key.txt")

        model = (custom_model or "").strip() or model

        content = []
        if prompt.strip():
            content.append({"type": "text", "text": prompt})

        if mode == "文生视频":
            if not prompt.strip():
                raise RuntimeError("文生视频必须写提示词")
        elif mode == "首帧图生视频":
            if image_first is None:
                raise RuntimeError("首帧图生视频需要接 image_first")
            content.append({"type": "image_url", "role": "first_frame",
                            "image_url": {"url": _tensor_to_data_uri(image_first)}})
        elif mode == "首尾帧图生视频":
            if image_first is None or image_last is None:
                raise RuntimeError("首尾帧生视频需要 image_first + image_last 都接上")
            content.append({"type": "image_url", "role": "first_frame",
                            "image_url": {"url": _tensor_to_data_uri(image_first)}})
            content.append({"type": "image_url", "role": "last_frame",
                            "image_url": {"url": _tensor_to_data_uri(image_last)}})
        else:  # 参考图生视频
            refs = [x for x in (ref_image1, ref_image2, ref_image3) if x is not None]
            if not refs:
                raise RuntimeError("参考图生视频至少要接一张 ref_image")
            for t in refs:
                content.append({"type": "image_url", "role": "reference_image",
                                "image_url": {"url": _tensor_to_data_uri(t)}})

        payload = {
            "model": model,
            "content": content,
            "resolution": resolution,
            "ratio": ratio,
            "generate_audio": bool(generate_audio),
            "watermark": bool(watermark),
        }
        if duration > 0:
            payload["duration"] = int(duration)
        if output_format:
            payload["output_format"] = output_format

        t0 = time.time()
        r = requests.post(TASKS_URL, headers={"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"},
                          json=payload, timeout=90)
        if r.status_code >= 400:
            raise RuntimeError(f"创建任务失败 {r.status_code}: {r.text[:800]}")
        task_id = r.json().get("id")
        if not task_id:
            raise RuntimeError("没拿到任务 id：" + r.text[:500])
        print(f"[火山方舟 Seedance] 任务已提交 {task_id}，等待生成…")

        deadline = time.time() + timeout
        status, video_url, extra = "queued", None, {}
        while time.time() < deadline:
            time.sleep(5)
            q = requests.get(f"{TASKS_URL}/{task_id}",
                             headers={"Authorization": "Bearer " + key}, timeout=60)
            if q.status_code >= 400:
                raise RuntimeError(f"查询任务失败 {q.status_code}: {q.text[:500]}")
            body = q.json()
            status = body.get("status", "")
            if status == "succeeded":
                video_url = (body.get("content") or {}).get("video_url")
                extra = body
                break
            if status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"任务 {status}: " + json.dumps(body, ensure_ascii=False)[:800])
            print(f"[火山方舟 Seedance] {task_id} -> {status} ({time.time() - t0:.0f}s)")

        if not video_url:
            raise RuntimeError(f"等待超时（{timeout}s），任务 {task_id} 仍未完成，可用任务 id 去控制台查")

        got = requests.get(video_url, timeout=300)
        got.raise_for_status()
        buf = io.BytesIO(got.content)
        video = InputImpl.VideoFromFile(buf)

        out_path = os.path.join(folder_paths.get_output_directory(),
                                f"seedance_{task_id}.{output_format or 'mp4'}")
        try:
            with open(out_path, "wb") as f:
                f.write(got.content)
        except OSError:
            out_path = "(保存失败)"

        info = (f"task={task_id}  model={model}  {resolution} {ratio} {duration}s "
                f"audio={generate_audio}  {len(got.content) / 1e6:.1f}MB  "
                f"{time.time() - t0:.0f}s  file={out_path}")
        print("[火山方舟 Seedance] " + info)

        # 尾帧给下一段当首帧用（分段拼接），拿不到也不影响主输出
        try:
            last_frame = _last_frame_tensor(got.content)
        except Exception as exc:  # noqa: BLE001
            print(f"[火山方舟 Seedance] 尾帧提取失败：{exc}")
            last_frame = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (video, info, last_frame)


NODE_CLASS_MAPPINGS = {"ArkSeedanceVideo": ArkSeedanceVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"ArkSeedanceVideo": "火山方舟 Seedance 视频"}
