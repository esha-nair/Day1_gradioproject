import os
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

def predict(message, history, system_prompt, temperature):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": entry["content"]})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            human, assistant = entry
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        stream = client.chat_completion(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=max(temperature, 0.01),
            stream=True
        )

        for chunk in stream:
            token = None
            try:
                choices = chunk.choices
                if choices and len(choices) > 0:
                    delta = choices[0].delta
                    if delta and hasattr(delta, "content"):
                        token = delta.content
            except (AttributeError, TypeError):
                pass

            if token is None:
                try:
                    choices = chunk.get("choices", [])
                    if choices and len(choices) > 0:
                        token = choices[0].get("delta", {}).get("content")
                except (AttributeError, TypeError):
                    pass

            if token:
                response += token
                yield response
                
    except Exception as e:
        yield f"An error occurred with the AI server: {str(e)}"

system_prompt_textbox = gr.Textbox(
    value="You are a helpful AI assistant.", 
    label="System Prompt (Define Personality)", 
    placeholder="e.g., Act like a pirate...",
    lines=2
)

temperature_slider = gr.Slider(
    minimum=0.0, 
    maximum=1.2, 
    value=0.7, 
    step=0.1, 
    label="Temperature (Randomness)"
)

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Dynamically Controlled Chatbot")
    gr.Markdown("Adjust the system prompt or temperature on the left to immediately change the chatbot's behavior.")
    
    with gr.Row():
        with gr.Column(scale=1):
            system_prompt_textbox.render()
            temperature_slider.render()
            
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=predict,
                additional_inputs=[system_prompt_textbox, temperature_slider]
            )

if __name__ == "__main__":
    demo.launch()