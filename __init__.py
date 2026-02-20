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
                        "tooltip": "Output the internal reasoning process of the model in the text_output.",
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
                    is_thought = getattr(part, "thought", False)
                    pil_output = getattr(part, "as_image", lambda: None)()
                    if is_thought and show_thoughts:
                        if part.text:
                            all_text_outputs.append(
                                f"<Thought>\n{part.text}\n</Thought>"
                            )
                        elif pil_output is not None:
                            all_text_outputs.append(
                                "<Thought>\n[Intermediate image generated]\n</Thought>"
                            )
                    elif part.text and not is_thought:
                        all_text_outputs.append(f"<Response>\n{part.text}\n</Response>")

                    if pil_output:
                        all_pil_outputs.append(pil_output)
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
                is_thought = getattr(part, "thought", False)
                pil_output = getattr(part, "as_image", lambda: None)()
                if is_thought and show_thoughts:
                    if part.text:
                        all_text_outputs.append(f"<Thought>\n{part.text}\n</Thought>")
                    elif pil_output is not None:
                        all_text_outputs.append(
                            "<Thought>\n[Intermediate image generated]\n</Thought>"
                        )
                elif part.text and not is_thought:
                    all_text_outputs.append(f"<Response>\n{part.text}\n</Response>")

                if pil_output:
                    all_pil_outputs.append(pil_output)

        updated_history = chat.get_history()

        if not all_pil_outputs:
            print("Nano Banana Warning: No image returned from API.")
            batch_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            tensor_outputs = [self._pil_to_tensor(img) for img in all_pil_outputs]
            batch_tensor = torch.cat(tensor_outputs, dim=0)

        final_text = "\n\n".join(all_text_outputs)

        return (batch_tensor, final_text, updated_history)


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
    "NanoBananaTextDisplay": NanoBananaTextDisplay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaAgent": "Nano Banana Agent",
    "NanoBananaTextDisplay": "Nano Banana Text Display",
}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
