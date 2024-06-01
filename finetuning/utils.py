import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import AddedToken
import numpy as np

def download_model(model_name_or_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    # Add chat template
    tokenizer.chat_template = """{% set loop_messages = messages %}{% if messages[0]['input'] ==''%}{% set system  = '<|start_header_id|>' + '<|system|>' + '<|end_header_id|>\n\n'+ "Act like you are an excellent coder and always come up with faithful and trustworthy answers. Below is an instruction that describes a task. Write a response that appropriately completes the request.Please remember that if the prompt is related to coding, you should provide a clear, concise, and accurate code example or explanation. Otherwise, if the prompt is a general question or a conversational statement, you should respond in a natural and friendly manner appropriate for casual conversation." + '<|eot_id|>' %}{% else %}{% set system  = '<|start_header_id|>' + '<|system|>' + '<|end_header_id|>\n\n'+ "Act like you are an excellent coder and always come up with faithful and trustworthy answers. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Please remember that if the prompt is related to coding, you should provide a clear, concise, and accurate code example or explanation. Otherwise, if the prompt is a general question or a conversational statement, you should respond in a natural and friendly manner appropriate for casual conversation." + '<|eot_id|>' %}{% endif%}{{bos_token + system}}{% for message in loop_messages %}{% set instruction = '<|start_header_id|>' + '<|instruction|>' + '<|end_header_id|>\n\n'+ message['instruction'] | trim +'<|eot_id|>' %}{% if 'output' not in message.keys()%}{% set output ='' %}{% else %}{% set output = '<|start_header_id|>' + '<|output|>' + '<|end_header_id|>\n\n' + message['output'] | trim + '<|eot_id|>' %}{% endif %}{% if message['input'] ==''%}{% set content = instruction + output %}{% else %}{% set input = '<|start_header_id|>' + '<|input|>' + '<|end_header_id|>\n\n' + message['input'] | trim + '<|eot_id|>' %}{% set content = instruction + input + output %}{% endif%}{{content}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|><|output|><|end_header_id|>\n\n' }}{% endif %}"""

    tokenizer.add_tokens([AddedToken("<|system|>"),
                          AddedToken("<|instruction|>"),
                          AddedToken("<|input|>"),
                          AddedToken("<|output|>")]),
    tokenizer.save_pretrained(output_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        token = "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"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(output_dir)
    
    print(f"Model and tokenizer are saved to {output_dir}")

def test_chat_template():
    tokenizer = AutoTokenizer.from_pretrained("/home/user/LLAMA3_for_coding/finetuning/Meta-Llama-3-8b")
    conversation = [{'output': 'arr = [2, 4, 6, 8, 10]',
        'instruction': 'Create an array of length 5 which contains all even numbers between 1 and 10.',
        'input': '',
        'language': 'Ruby'},
        {
        'instruction': 'Create a python list in python','input': ''}]

    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    print(tokenizer.convert_ids_to_tokens(ids[0]))
    print(text)

def merge_lora(
    base_model_path: str = "/home/user/LLAMA3_for_coding/finetuning/Meta-Llama-3-8b",
    lora_path: str = 'unsloth-lora-checkpoints-r256a512-31_5_11am/checkpoint-1223',
    output_path: str = 'llama3-8b-lora-chat_1'
    ):
    import huggingface_hub
    huggingface_hub.login(token = "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")
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
    )
    lora_model = PeftModel.from_pretrained(base, lora_path)
    model = lora_model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    # model.push_to_hub("dinhlnd1610/LLAMA3-8B-Coding")
    # tokenizer.push_to_hub("dinhlnd1610/LLAMA3-8B-Coding")
    
def quantize_vector(num_bits):
    # A random vector x
    x = np.array([0.2, 0.7, 0.4, 0.9, 0.1])
    # Determine the range of the vector
    x_min = np.min(x)
    x_max = np.max(x)
    # Calculate the step size based on the number of bits
    step_size = (x_max - x_min) / (2 ** num_bits - 1)
    # Quantize the vector
    x_quantized = np.round((x - x_min) / step_size)
    return x_quantized

def dequantize_vector():
    x_quantized = np.array([2, 11, 6, 15, 0])
    # x_quantized = np.array([32, 191, 96, 255, 0])
    x_min, x_max, num_bits = 0.1, 0.9, 8
    # Calculate the step size based on the number of bits
    step_size = (x_max - x_min) / (2 ** num_bits - 1)
    # Dequantize the vector
    x_dequantized = x_quantized * step_size + x_min
    return x_dequantized

if __name__ == "__main__":
    fire.Fire()

#https://www.reddit.com/r/LocalLLaMA/comments/1ae0uig/how_many_epochs_do_you_train_an_llm_for_in_the/