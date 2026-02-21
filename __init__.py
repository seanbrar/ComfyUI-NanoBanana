import os
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types


class NanoBananaAgent:
    """A unified ComfyUI Node for Nano Banana API generation and chat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "Enter API Key or use GEMINI_API_KEY env",
                        "multiline": False,
                        "tooltip": "Your Gemini API Key. Can also be set via the GEMINI_API_KEY environment variable.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A highly detailed portrait...",
                        "tooltip": "The text description of the image you want to generate or edit.",
                    },
                ),
                "use_pro": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Pro",
                        "label_off": "Flash",
                        "tooltip": "Use gemini-3-pro-image-preview (highest quality, reasoning) vs gemini-2.5-flash-image (fast, cheaper).",
                    },
                ),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4"],
                    {
                        "default": "1:1",
                        "tooltip": "The dimensional ratio of the output image.",
                    },
                ),
                "resolution": (
                    ["None", "1K", "2K", "4K"],
                    {
                        "default": "None",
                        "tooltip": "(Pro Only) The target resolution scale of the image.",
                    },
                ),
                "batch_mode": (
                    ["combine", "individual"],
                    {
                        "default": "combine",
                        "tooltip": "If passing multiple reference images: 'combine' sends them all in one prompt. 'individual' generates a separate result for each image. (Note: ignored if chat_history is connected)",
                    },
                ),
                "enable_search": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Enabled",
                        "label_off": "Disabled",
                        "tooltip": "Allow the model to use Google Search to ground its generation with real-world facts.",
                    },
                ),
                "show_thoughts": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Enabled",
                        "label_off": "Disabled",
                        "tooltip": "Output the internal reasoning process of the model in the text_output. Only supported with the Pro model; ignored when using Flash.",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional starting image or visual context for editing/generation."
                    },
                ),
                "chat_history": (
                    "NANO_CHAT_HISTORY",
                    {
                        "tooltip": "Connect the history output from a previous turn to continue a conversation."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "NANO_CHAT_HISTORY")
    RETURN_NAMES = ("image", "text_output", "chat_history")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"
    DESCRIPTION = "Generates or edits images using the Nano Banana Google API, optionally maintaining a conversational history."

    def _get_api_key(self, api_key_input: str) -> str:
        key = (
            os.environ.get("GEMINI_API_KEY")
            if "GEMINI_API_KEY" not in api_key_input
            and api_key_input == "Enter API Key or use GEMINI_API_KEY env"
            else api_key_input
        )
        if not key or key == "Enter API Key or use GEMINI_API_KEY env":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "Nano Banana: API Key is required. Please provide it in the node or set the GEMINI_API_KEY environment variable."
                )
        return key

    def _pil_to_tensor(self, img) -> torch.Tensor:
        if hasattr(img, "image_bytes") and getattr(img, "image_bytes", None):
            import io

            img = Image.open(io.BytesIO(img.image_bytes))

        image_tensor = img.convert("RGB")
        image_tensor = np.array(image_tensor).astype(np.float32) / 255.0
        image_tensor = np.clip(image_tensor, 0.0, 1.0)
        return torch.from_numpy(image_tensor)[None,]

    def generate(
        self,
        api_key,
        prompt,
        use_pro,
        aspect_ratio,
        resolution,
        batch_mode,
        enable_search,
        show_thoughts,
        reference_image=None,
        chat_history=None,
    ):
        key = self._get_api_key(api_key)
        client = genai.Client(api_key=key)
        model = "gemini-3-pro-image-preview" if use_pro else "gemini-2.5-flash-image"

        config_options = {"response_modalities": ["TEXT", "IMAGE"]}
        image_config = {"aspect_ratio": aspect_ratio}

        if resolution != "None":
            image_config["image_size"] = resolution

        config_options["image_config"] = types.ImageConfig(**image_config)

        if enable_search:
            config_options["tools"] = [{"google_search": {}}]

        if show_thoughts:
            if use_pro:
                config_options["thinking_config"] = {"include_thoughts": True}
            else:
                print(
                    "Nano Banana Warning: 'show_thoughts' is only supported with the Pro model. Ignoring for Flash."
                )

        config = types.GenerateContentConfig(**config_options)

        history = chat_history if chat_history is not None else []
        chat = client.chats.create(model=model, config=config, history=history)

        actual_batch_mode = batch_mode
        if (
            chat_history is not None
            and len(chat_history) > 0
            and reference_image is not None
        ):
            if batch_mode == "individual":
                print(
                    "Nano Banana Warning: 'individual' batch_mode is ignored because a chat_history is connected. Combining images into a single chat turn."
                )
                actual_batch_mode = "combine"

        all_pil_outputs = []
        all_text_outputs = []

        if reference_image is not None and actual_batch_mode == "individual":
            for img_tensor in reference_image:
                current_turn = [prompt]
                img_np = img_tensor.cpu().numpy() * 255.0
                pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                current_turn.append(pil_img)

                try:
                    response = chat.send_message(current_turn)
                except Exception as e:
                    print(f"Nano Banana API Error: {e}")
                    raise ValueError(f"Nano Banana API call failed: {e}")

                for part in response.parts:
                    try:
                        if part.thought:
                            if show_thoughts:
                                if part.text:
                                    all_text_outputs.append(
                                        f"<Thought>\n{part.text}\n</Thought>"
                                    )
                                else:
                                    pil_output = part.as_image()
                                    if pil_output:
                                        all_text_outputs.append(
                                            "<Thought>\n[Intermediate image generated]\n</Thought>"
                                        )
                                        all_pil_outputs.append(pil_output)
                        else:
                            if part.text:
                                all_text_outputs.append(
                                    f"<Response>\n{part.text}\n</Response>"
                                )
                            else:
                                pil_output = part.as_image()
                                if pil_output:
                                    all_pil_outputs.append(pil_output)
                    except Exception as e:
                        print(f"NanoBanana warning during part parsing: {e}")
        else:
            current_turn = [prompt]
            if reference_image is not None:
                for img_tensor in reference_image:
                    img_np = img_tensor.cpu().numpy() * 255.0
                    pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    current_turn.append(pil_img)

            try:
                response = chat.send_message(current_turn)
            except Exception as e:
                print(f"Nano Banana API Error: {e}")
                raise ValueError(f"Nano Banana API call failed: {e}")

            for part in response.parts:
                try:
                    if part.thought:
                        if show_thoughts:
                            if part.text:
                                all_text_outputs.append(
                                    f"<Thought>\n{part.text}\n</Thought>"
                                )
                            else:
                                pil_output = part.as_image()
                                if pil_output:
                                    all_text_outputs.append(
                                        "<Thought>\n[Intermediate image generated]\n</Thought>"
                                    )
                                    all_pil_outputs.append(pil_output)
                    else:
                        if part.text:
                            all_text_outputs.append(
                                f"<Response>\n{part.text}\n</Response>"
                            )
                        else:
                            pil_output = part.as_image()
                            if pil_output:
                                all_pil_outputs.append(pil_output)
                except Exception as e:
                    print(f"NanoBanana warning during part parsing: {e}")

        updated_history = chat.get_history()

        if not all_pil_outputs:
            print("Nano Banana Warning: No image returned from API.")
            batch_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            tensor_outputs = [self._pil_to_tensor(img) for img in all_pil_outputs]
            batch_tensor = torch.cat(tensor_outputs, dim=0)

        final_text = "\n\n".join(all_text_outputs)

        return (batch_tensor, final_text, updated_history)


class NanoBananaLLM:
    """A ComfyUI Node for text-only LLM generation using the Gemini API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "Enter API Key or use GEMINI_API_KEY env",
                        "multiline": False,
                        "tooltip": "Your Gemini API Key. Can also be set via the GEMINI_API_KEY environment variable.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant.",
                        "tooltip": "The system instruction that defines the LLM's behavior.",
                    },
                ),
                "user_input": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "The primary text input for the LLM. Can be typed directly or wired from another node.",
                    },
                ),
                "model": (
                    ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-pro-preview"],
                    {
                        "default": "gemini-2.5-flash",
                        "tooltip": "The Gemini model to use for text generation.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values are more deterministic, higher values are more creative.",
                    },
                ),
            },
            "optional": {
                "context": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional context wired from an upstream node. Prepended to user_input.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "Nano Banana"
    DESCRIPTION = "Generates text using the Gemini API. Useful for classification, prompt engineering, and other text-only LLM tasks."

    def _get_api_key(self, api_key_input: str) -> str:
        key = (
            os.environ.get("GEMINI_API_KEY")
            if "GEMINI_API_KEY" not in api_key_input
            and api_key_input == "Enter API Key or use GEMINI_API_KEY env"
            else api_key_input
        )
        if not key or key == "Enter API Key or use GEMINI_API_KEY env":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "Nano Banana: API Key is required. Please provide it in the node or set the GEMINI_API_KEY environment variable."
                )
        return key

    def generate_text(self, api_key, system_prompt, user_input, model, temperature, context=None):
        key = self._get_api_key(api_key)
        client = genai.Client(api_key=key)

        if context and context.strip():
            combined_input = f"{context.strip()}\n\n{user_input}"
        else:
            combined_input = user_input

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_modalities=["TEXT"],
            temperature=temperature,
        )

        try:
            response = client.models.generate_content(
                model=model, contents=combined_input, config=config,
            )
        except Exception as e:
            raise ValueError(f"Nano Banana LLM API call failed: {e}")

        return (response.text or "",)


class NanoBananaLoadText:
    """A ComfyUI Node to load text content from a file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute or relative path to a .txt or .md file.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_text"
    CATEGORY = "Nano Banana"
    DESCRIPTION = "Loads text content from a file on disk. Supports .txt, .md, and other plain text formats."

    def load_text(self, file_path):
        path = os.path.expanduser(file_path.strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Nano Banana Load Text: File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return (text,)


class NanoBananaTextRouter:
    """A ComfyUI Node that maps a category string to a corresponding value."""

    CATEGORIES = ["Projects", "Reflections", "Media", "Logic", "Journaling"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "The category string from a categorizer node.",
                    },
                ),
                "projects": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value to output when category matches 'Projects'.",
                    },
                ),
                "reflections": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value to output when category matches 'Reflections'.",
                    },
                ),
                "media": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value to output when category matches 'Media'.",
                    },
                ),
                "logic": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value to output when category matches 'Logic'.",
                    },
                ),
                "journaling": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Value to output when category matches 'Journaling'.",
                    },
                ),
                "default": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Fallback value when category doesn't match any option.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "route"
    CATEGORY = "Nano Banana"
    DESCRIPTION = "Maps a category string (Projects, Reflections, Media, Logic, Journaling) to a corresponding value. Useful for routing LoRA names, style keywords, or other per-category settings."

    def route(self, category, projects, reflections, media, logic, journaling, default):
        lookup = {
            "projects": projects,
            "reflections": reflections,
            "media": media,
            "logic": logic,
            "journaling": journaling,
        }
        normalized = category.strip().lower()
        value = lookup.get(normalized, default)
        if not value:
            value = default
        return (value,)


class NanoBananaTextDisplay:
    """A ComfyUI Node to display text directly on the canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the text_output from the Nano Banana Agent here.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_text"
    CATEGORY = "Nano Banana"
    OUTPUT_NODE = True
    DESCRIPTION = "Displays multiline text directly on the node UI in ComfyUI."

    def display_text(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "NanoBananaAgent": NanoBananaAgent,
    "NanoBananaLLM": NanoBananaLLM,
    "NanoBananaLoadText": NanoBananaLoadText,
    "NanoBananaTextRouter": NanoBananaTextRouter,
    "NanoBananaTextDisplay": NanoBananaTextDisplay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAgent": "Nano Banana Agent",
    "NanoBananaLLM": "Nano Banana LLM",
    "NanoBananaLoadText": "Nano Banana Load Text",
    "NanoBananaTextRouter": "Nano Banana Text Router",
    "NanoBananaTextDisplay": "Nano Banana Text Display",
}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
