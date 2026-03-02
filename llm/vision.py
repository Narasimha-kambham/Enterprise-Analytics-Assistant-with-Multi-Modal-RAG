import os
import json
import base64
from pathlib import Path
from langchain.messages import HumanMessage
from llm.gemini_client import get_gemini_model

from config.settings import VISION_CACHE_PATH

CACHE_FILE = VISION_CACHE_PATH


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_description(image_path: str) -> str:
    image_path = str(Path(image_path).resolve())  # Absolute path for stable caching
    cache = load_cache()

    if image_path in cache:
        return cache[image_path]

    try:
        model = get_gemini_model()

        image_b64 = image_to_base64(image_path)

        # Detect correct MIME type
        suffix = Path(image_path).suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"

        data_uri = f"data:{mime_type};base64,{image_b64}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this chart in structured financial format."},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ]
        )

        response = model.invoke([message])
        description = response.content

        cache[image_path] = description
        save_cache(cache)

        return description

    except Exception as e:
        print(f"Vision processing skipped for {image_path}: {e}")
        return "Vision processing unavailable."