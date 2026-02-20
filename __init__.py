import os
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types


class NanoBananaGenerator:
    """A ComfyUI Node wrapping your Nano Banana API logic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "Enter API Key or use GEMINI_API_KEY env",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A highly detailed portrait..."},
                ),
                "use_pro": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Pro", "label_off": "Flash"},
                ),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4"],
                    {"default": "1:1"},
                ),
                "resolution": (["None", "1K", "2K", "4K"], {"default": "None"}),
                "batch_mode": (
                    ["combine", "individual"],
                    {"default": "combine"},
                ),
                "enable_search": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enabled", "label_off": "Disabled"},
                ),
                "show_thoughts": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enabled", "label_off": "Disabled"},
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text_output")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"

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

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        image_tensor = img.convert("RGB")
        image_tensor = np.array(image_tensor).astype(np.float32) / 255.0
        image_tensor = np.clip(image_tensor, 0.0, 1.0)
        return torch.from_numpy(image_tensor)[None,]

    def _call_api(self, client, model, contents, config, show_thoughts):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            print(f"Nano Banana API Error: {e}")
            raise ValueError(f"Nano Banana API call failed: {e}")

        pil_outputs = []
        text_outputs = []

        for part in response.parts:
            is_thought = getattr(part, "thought", False)
            if is_thought and show_thoughts:
                if part.text:
                    text_outputs.append(f"Thought: {part.text}")
                elif getattr(part, "inline_data", None):
                    text_outputs.append("Thought: [Intermediate image generated]")
            elif part.text and not is_thought:
                text_outputs.append(f"Output: {part.text}")

            pil_output = getattr(part, "as_image", lambda: None)()
            if pil_output:
                pil_outputs.append(pil_output)

        if not pil_outputs:
            print(
                f"Nano Banana Warning: API returned empty image for contents: {contents}"
            )

        return pil_outputs, text_outputs

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

        all_pil_outputs = []
        all_text_outputs = []

        if reference_image is not None and batch_mode == "individual":
            for img_tensor in reference_image:
                contents = [prompt]
                img_np = img_tensor.cpu().numpy() * 255.0
                pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                contents.append(pil_img)

                pil_outs, text_outs = self._call_api(
                    client, model, contents, config, show_thoughts
                )
                all_pil_outputs.extend(pil_outs)
                all_text_outputs.extend(text_outs)
        else:
            contents = [prompt]
            if reference_image is not None:
                for img_tensor in reference_image:
                    img_np = img_tensor.cpu().numpy() * 255.0
                    pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    contents.append(pil_img)

            pil_outs, text_outs = self._call_api(
                client, model, contents, config, show_thoughts
            )
            all_pil_outputs.extend(pil_outs)
            all_text_outputs.extend(text_outs)

        if not all_pil_outputs:
            raise ValueError("Nano Banana: No image returned from API")

        tensor_outputs = [self._pil_to_tensor(img) for img in all_pil_outputs]
        batch_tensor = torch.cat(tensor_outputs, dim=0)
        final_text = "\n\n".join(all_text_outputs)

        return (batch_tensor, final_text)


class NanoBananaChat:
    """A ComfyUI Node for stateful Nano Banana Chat sessions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "Enter API Key or use GEMINI_API_KEY env",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Refine the image by..."},
                ),
                "use_pro": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Pro", "label_off": "Flash"},
                ),
                "enable_search": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enabled", "label_off": "Disabled"},
                ),
                "show_thoughts": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enabled", "label_off": "Disabled"},
                ),
            },
            "optional": {
                "chat_history": ("NANO_CHAT_HISTORY",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "NANO_CHAT_HISTORY")
    RETURN_NAMES = ("image", "text_output", "chat_history")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"

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
                raise ValueError("Nano Banana: API Key is required.")
        return key

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        image_tensor = img.convert("RGB")
        image_tensor = np.array(image_tensor).astype(np.float32) / 255.0
        image_tensor = np.clip(image_tensor, 0.0, 1.0)
        return torch.from_numpy(image_tensor)[None,]

    def generate(
        self,
        api_key,
        prompt,
        use_pro,
        enable_search,
        show_thoughts,
        chat_history=None,
        reference_image=None,
    ):
        key = self._get_api_key(api_key)
        client = genai.Client(api_key=key)
        model = "gemini-3-pro-image-preview" if use_pro else "gemini-2.5-flash-image"

        contents = chat_history.copy() if chat_history is not None else []

        current_turn = [prompt]
        if reference_image is not None:
            for img_tensor in reference_image:
                img_np = img_tensor.cpu().numpy() * 255.0
                pil_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                current_turn.append(pil_img)

        contents.extend(current_turn)

        config_options = {"response_modalities": ["TEXT", "IMAGE"]}
        if enable_search:
            config_options["tools"] = [{"google_search": {}}]

        config = types.GenerateContentConfig(**config_options)

        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            print(f"Nano Banana API Error in Chat: {e}")
            raise ValueError(f"Nano Banana API call failed in Chat: {e}")

        pil_outputs = []
        text_outputs = []

        assistant_turn = []

        for part in response.parts:
            is_thought = getattr(part, "thought", False)
            if is_thought and show_thoughts:
                if part.text:
                    text_outputs.append(f"Thought: {part.text}")
                elif getattr(part, "inline_data", None):
                    text_outputs.append("Thought: [Intermediate image generated]")
            elif part.text and not is_thought:
                text_outputs.append(f"Output: {part.text}")
                assistant_turn.append(part.text)

            pil_output = getattr(part, "as_image", lambda: None)()
            if pil_output:
                pil_outputs.append(pil_output)
                assistant_turn.append(pil_output)

        if not pil_outputs:
            print("Nano Banana: No image returned from API in Chat turn.")

        contents.extend(assistant_turn)

        if pil_outputs:
            tensor_outputs = [self._pil_to_tensor(img) for img in pil_outputs]
            batch_tensor = torch.cat(tensor_outputs, dim=0)
        else:
            batch_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        final_text = "\n\n".join(text_outputs)

        return (batch_tensor, final_text, contents)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "NanoBananaGenerator": NanoBananaGenerator,
    "NanoBananaChat": NanoBananaChat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGenerator": "Nano Banana API Generator",
    "NanoBananaChat": "Nano Banana API Chat",
}
