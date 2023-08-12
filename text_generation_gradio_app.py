# -*- coding: utf-8 -*-
"""Text-Generation-Gradio-App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OfP8zY_Nwx2U2QeYnYRanlH7_SuKzmGq
"""
# %pip install -q gradio
# %pip install -q git+https://github.com/huggingface/transformers.git
import gradio as gr
import tensorflow as tf

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)

def generate_text(inp):
    input_ids = tokenizer.encode(inp, return_tensors='tf')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping= True)
    output = tokenizer.decode(beam_output[0], skip_special_token=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

output_text = gr.outputs.Textbox()
gr.Interface(generate_text,"textbox",output_text,title="Text Generation machine ",description="Ask any question. Note: It can take 20-60 seconds to generate output based on your internet connection.").launch()