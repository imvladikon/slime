"""Training processor loader for GLM-5.3-Flash on pre-5.16 Transformers."""

from __future__ import annotations

import json
from pathlib import Path

_MEDIA_REMINDER = (
    '{{- "<reminder>You are unable to process this " ~ media_type ~ '
    '" because you don\'t have multi-modal input ability. Try different methods.</reminder>" }}'
)
_IMAGE_PLACEHOLDER = """{%- if media_type == 'image' -%}
                {{- "<|begin_of_image|><|image|><|end_of_image|>" }}
                {%- else -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}
                {%- endif -%}"""


def is_glm5_next_checkpoint(name_or_path: str) -> bool:
    config_path = Path(name_or_path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as stream:
        return json.load(stream).get("model_type") == "glm5_next"


def _chat_template(checkpoint: Path) -> str | None:
    path = checkpoint / "chat_template.jinja"
    if not path.is_file():
        return None
    template = path.read_text(encoding="utf-8")
    if "<|image|>" in template:
        return template
    if _MEDIA_REMINDER not in template:
        raise ValueError("GLM-5.3 chat template has no recognized media rendering branch")
    return template.replace(_MEDIA_REMINDER, _IMAGE_PLACEHOLDER, 1)


def load_glm5_next_processor(name_or_path: str, **tokenizer_kwargs):
    """Build the official image processor without depending on AutoProcessor registration."""
    from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
    from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor

    from slime.utils.processing_utils import load_tokenizer
    from slime_plugins.models.glm5_next.image_processing import Glm5NextImageProcessor

    checkpoint = Path(name_or_path)
    processor_config_path = checkpoint / "processor_config.json"
    processor_config = {}
    if processor_config_path.is_file():
        with processor_config_path.open(encoding="utf-8") as stream:
            processor_config = json.load(stream)
    image_config = dict(processor_config.get("image_processor", {}))
    image_config.pop("image_processor_type", None)
    video_config = dict(processor_config.get("video_processor", {}))
    video_config.pop("video_processor_type", None)
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    tokenizer = load_tokenizer(name_or_path, **tokenizer_kwargs)
    return Glm46VProcessor(
        image_processor=Glm5NextImageProcessor(**image_config),
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor(**video_config),
        chat_template=_chat_template(checkpoint) or getattr(tokenizer, "chat_template", None),
    )


__all__ = ["is_glm5_next_checkpoint", "load_glm5_next_processor"]

