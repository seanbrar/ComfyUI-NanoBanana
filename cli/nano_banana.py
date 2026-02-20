"""
Nano Banana API Wrapper
Integrates Gemini 2.5 Flash Image and Gemini 3 Pro Image Preview features.
Supports text-to-image, image editing, chat sessions, search grounding,
and high-resolution outputs.
"""

import argparse
import os
from typing import List, Optional, Union

from PIL import Image
from google import genai
from google.genai import types


class NanoBananaClient:
    """Provides a minimalist interface to Nano Banana image models."""

    FLASH_MODEL = "gemini-2.5-flash-image"
    PRO_MODEL = "gemini-3-pro-image-preview"

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _process_response(
        self, response: types.GenerateContentResponse, output_prefix: str, show_thoughts: bool
    ):
        """Extracts text, thoughts, and images from the API response."""
        image_index = 0
        for part in response.parts:
            is_thought = getattr(part, "thought", False)

            if is_thought and show_thoughts:
                if part.text:
                    print(f"Thought: {part.text}")
                elif getattr(part, "inline_data", None):
                    print("Thought: [Intermediate image generated]")

            if part.text and not is_thought:
                print(f"Output: {part.text}")

            image = getattr(part, "as_image", lambda: None)()
            if image:
                suffix = f"_{image_index}" if image_index > 0 else ""
                filename = f"{output_prefix}{suffix}.png"
                image.save(filename)
                print(f"Saved: {filename}")
                image_index += 1

    def generate(
        self,
        prompt: str,
        output_prefix: str = "output",
        reference_images: Optional[List[Union[str, Image.Image]]] = None,
        use_pro: bool = False,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        enable_search: bool = False,
        show_thoughts: bool = False,
    ):
        """Generates or edits images with optional configurations and context."""
        model = self.PRO_MODEL if use_pro else self.FLASH_MODEL
        contents = [prompt]

        if reference_images:
            for img in reference_images:
                contents.append(Image.open(img) if isinstance(img, str) else img)

        config_options = {"response_modalities": ["TEXT", "IMAGE"]}
        image_config = {}

        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if resolution:
            image_config["image_size"] = resolution

        if image_config:
            config_options["image_config"] = types.ImageConfig(**image_config)

        if enable_search:
            config_options["tools"] = [{"google_search": {}}]

        config = types.GenerateContentConfig(**config_options)

        response = self.client.models.generate_content(
            model=model, contents=contents, config=config
        )

        self._process_response(response, output_prefix, show_thoughts)

    def create_chat(self, enable_search: bool = False) -> genai.chats.Chat:
        """Initializes a conversational session for iterative editing."""
        config_options = {"response_modalities": ["TEXT", "IMAGE"]}
        if enable_search:
            config_options["tools"] = [{"google_search": {}}]

        config = types.GenerateContentConfig(**config_options)

        return self.client.chats.create(model=self.PRO_MODEL, config=config)


def main():
    parser = argparse.ArgumentParser(description="Nano Banana Image Generation CLI")
    parser.add_argument("--api-key", help="Gemini API Key. Falls back to GEMINI_API_KEY env var.")
    parser.add_argument("--prompt", required=True, help="Text instructions for generation or editing.")
    parser.add_argument("--output", default="output", help="Base filename for generated images.")
    parser.add_argument("--pro", action="store_true", help="Use the Nano Banana Pro model.")
    parser.add_argument("--search", action="store_true", help="Enable Google Search Grounding.")
    parser.add_argument("--thoughts", action="store_true", help="Display intermediate thinking steps.")
    parser.add_argument("--aspect-ratio", help="Outputs aspect ratio (e.g., '16:9', '1:1').")
    parser.add_argument("--resolution", choices=["1K", "2K", "4K"], help="Output resolution (Pro only).")
    parser.add_argument("--refs", nargs="+", help="Paths to reference images (up to 14 for Pro).")

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        parser.error("API key is required. Pass --api-key or set GEMINI_API_KEY.")

    client = NanoBananaClient(api_key=api_key)
    client.generate(
        prompt=args.prompt,
        output_prefix=args.output,
        reference_images=args.refs,
        use_pro=args.pro,
        aspect_ratio=args.aspect_ratio,
        resolution=args.resolution,
        enable_search=args.search,
        show_thoughts=args.thoughts
    )


if __name__ == "__main__":
    main()
