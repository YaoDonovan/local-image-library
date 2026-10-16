"""Local SigLIP image/text embeddings. Model files are loaded only from disk."""
from __future__ import annotations

import os
import threading
from pathlib import Path

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_VERSION = "siglip2-base-patch16-224@75de2d55"
VECTOR_SIZE = 768
BATCH_SIZE = 4


class SiglipEngine:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.lock = threading.RLock()
        self._torch = None
        self._processor = None
        self._model = None
        self.device = "unloaded"

    @property
    def available(self) -> bool:
        required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
        return self.model_dir.is_dir() and all((self.model_dir / name).is_file() for name in required)

    def load(self) -> None:
        with self.lock:
            if self._model is not None:
                return
            if not self.available:
                raise RuntimeError(f"未找到完整的本地 SigLIP2 模型：{self.model_dir}")
            import torch
            import transformers
            from transformers import AutoModel, AutoProcessor

            self._torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
            dtype_key = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True, **{dtype_key: dtype}
            ).eval().to(self.device)

    def _to_device(self, inputs: dict) -> dict:
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def encode_images(self, images: list) -> "np.ndarray":
        import numpy as np

        with self.lock:
            self.load()
            inputs = self._to_device(self._processor(images=images, return_tensors="pt"))
            with self._torch.inference_mode():
                result = self._model.get_image_features(**inputs).pooler_output
                result = self._torch.nn.functional.normalize(result.float(), dim=-1)
            return result.cpu().numpy().astype(np.float16)

    def encode_text(self, text: str) -> "np.ndarray":
        import numpy as np

        with self.lock:
            self.load()
            inputs = self._to_device(self._processor(
                text=[text], padding="max_length", truncation=True, return_tensors="pt"
            ))
            with self._torch.inference_mode():
                result = self._model.get_text_features(**inputs).pooler_output
                result = self._torch.nn.functional.normalize(result.float(), dim=-1)
            return result[0].cpu().numpy().astype(np.float32)


def model_directory(app_dir: Path, data_dir: Path) -> Path:
    override = os.environ.get("LOCAL_IMAGE_LIBRARY_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    candidates = (
        app_dir / ".local-models" / "siglip2-base-patch16-224",
        data_dir / "models" / "siglip2-base-patch16-224",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])
