from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from client import call_api

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def on_submit(user_input, chatbot, temperature):
    text = (user_input or {}).get("text") or ""
    files = (user_input or {}).get("files") or []

    if not text.strip() and not files:
        return chatbot, gr.MultimodalTextbox(value=None)
    
    new_history = list(chatbot)
    image = None
    if files:
        image_path = files[0]
        image = Image.open(image_path)
        new_history.append({"role": "user", "content": {"path": image_path}})
    if text.strip():
        new_history.append({"role": "user", "content": text.strip()})

    try:
        reply, elapsed_ms = call_api(
            API_URL,
            text.strip(),
            image=image,
            temperature=temperature
        )
        assistant_text = f"{reply}\n\n*Generated in {elapsed_ms} ms*"
    except Exception as e:
        assistant_text = f"⚠️ API error: {e}"
    
    new_history.append({"role": "assistant", "content": assistant_text})

    return new_history, gr.MultimodalTextbox(value=None)


def on_clear():
    return []


with gr.Blocks(title="VLM Chatbot") as demo:
    gr.Markdown(
        "# VLM Chatbot\n"
        f"Model: **Customized VLM** · API: `{API_URL}`"
    )

    chatbot = gr.Chatbot(
        type="messages",
        height=500,
        label="Chat",
    )

    user_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="single",
        file_types=["image"],
        placeholder="Type a message and/or attach an image...",
        show_label=False,
    )

    with gr.Row():
        temperature = gr.Slider(
            minimum=0.0, maximum=1.5, value=0.7, step=0.05, label="Temperature"
        )
       
    clear_btn = gr.Button("Clear history", variant="secondary")

    user_input.submit(
        on_submit,
        inputs=[user_input, chatbot, temperature],
        outputs=[chatbot, user_input],
    )
    clear_btn.click(on_clear, outputs=[chatbot])


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
