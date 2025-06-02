import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "sshleifer/tiny-gpt2"
CHECKPOINT = "Kaushik/grpo-rlhf-demo-adapter" # upload LoRA weights here

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, CHECKPOINT).eval().cuda()

def chat(prompt, max_new_tokens=50):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    gen = model.generate(ids, max_new_tokens=max_new_tokens)
    return tok.decode(gen[0], skip_special_tokens=True)

demo = gr.Interface(fn=chat,
                    inputs=[gr.Textbox(lines=3, label="Prompt"),
                            gr.Slider(10, 200, 50, label="Max tokens")],
                    outputs="text",
                    title="Group Relative Policy Optimized GPT-2")
demo.launch()
