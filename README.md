# ComfyUI Nano Banana

A minimalist, high-powered ComfyUI integration for the **Google Gemini Nano Banana API** (supporting `gemini-2.5-flash-image` and `gemini-3-pro-image-preview`). 

This custom node pack allows you to generate images, apply real-world search grounding, execute image-to-image workflows, and—uniquely—view the model's internal "Chain of Thought" reasoning directly on your canvas.

---

## 🚀 Features

*   **Unified Agent Architecture:** A single `Nano Banana Agent` node handles both one-shot generations and multi-turn iterative chat. No need to juggle different node types.
*   **Built-in Text Display:** Includes a native `Nano Banana Text Display` node so you can read the model's text outputs and reasoning processes without installing third-party UI packs.
*   **Pro Model Support:** Toggle between the lighting-fast `gemini-2.5-flash-image` and the reasoning-capable `gemini-3-pro-image-preview` at the click of a button.
*   **Search Grounding:** Enable `google_search` tools to let the model pull real-world facts into its image generation process.
*   **Chain of Thought Visibility:** Enable `show_thoughts` (Pro model only) to see exactly how the model reasoned through your prompt before generating the final image.

---

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/SeanBrar/ComfyUI-NanoBanana.git
   ```
3. Install the required Google GenAI SDK:
   ```bash
   # Assuming you are using the embedded ComfyUI python
   ../../../../python_embeded/python.exe -m pip install google-genai
   ```
   *(Or just `pip install google-genai` if using a standard Python environment).*

---

## 🔑 Authentication

You must provide a Gemini API Key. You can do this in two ways:

1.  **(Recommended)** Set an environment variable on your system: 
    `export GEMINI_API_KEY="your_api_key_here"`
    The node will automatically detect this and you can leave the node's `api_key` field at its default text.
2.  Paste the API key directly into the `api_key` text input on the `Nano Banana Agent` node. 

---

## 🧠 Using the Nodes

You will find two nodes under the **Nano Banana** category in ComfyUI.

### 1. Nano Banana Agent
This is the core workhorse. 
*   **One-Shot Generation:** Type a prompt, choose your model and aspect ratio, and hit Queue.
*   **Iterative Chat:** To refine an image (e.g., "now make it nighttime"), take the `chat_history` output from your first node and plug it into the `chat_history` input of a second `Nano Banana Agent` node. 
*   **Image-to-Image:** Plug any ComfyUI image into the `reference_image` input.

### 2. Nano Banana Text Display
Connect the `text_output` (String) from the Agent node into this display node. It automatically creates a multiline text box on your canvas showing exactly what the model said, including its `<Thought>` blocks if `show_thoughts` is enabled.

---

## ❓ FAQ & Troubleshooting

### Why is my `batch_mode` setting being ignored?
If you hover over the `batch_mode` tooltip, it notes that this setting is **ignored if `chat_history` is connected**. 
If you pass 4 reference images to the Agent *and* it is in the middle of a chat conversation, creating 4 parallel timelines would break the linear chat history. The node safely overrides this and combines all reference images into the current single chat turn.

### The model isn't outputting any thoughts!
Two things are required to see thoughts:
1.  You must have the **Pro** model selected (`gemini-3-pro-image-preview`). The Flash model does not support thinking.
2.  You must toggle `show_thoughts` to **Enabled**. 

### How do I use the Chat functionality without losing my settings?
The unified architecture means every single `Nano Banana Agent` node in your chain acts as its own configuration point. If you connect a `chat_history` wire to a new node, the conversation continues, but the *new* node's aspect ratio, resolution, and model settings will apply to that specific turn. 

### Does it support multiple reference images?
Yes! You can use a standard ComfyUI `Image Batch` node to combine multiple images and pipe them into the `reference_image` input. 
