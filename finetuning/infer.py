import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'dinhlnd1610/LLAMA3-8B-Coding'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    use_cache=True,
)
# tokenizer.chat_template = """{% set loop_messages = messages %}{% if messages[0]['input'] ==''%}{% set system  = '<|start_header_id|>' + '<|system|>' + '<|end_header_id|>\n\n'+ "Act like you are an excellent coder and always come up with faithful and trustworthy answers. Below is an instruction that describes a task. Write a response that appropriately completes the request.Please remember that if the prompt is related to coding, you should provide a clear, concise, and accurate code example or explanation. Otherwise, if the prompt is a general question or a conversational statement, you should respond in a natural and friendly manner appropriate for casual conversation. Moreover, if you are in multi-turn conversations, you should maintain context and coherence throughout the dialogue, ensuring continuity and relevance in responses. For example:\nExample 1: Coding-related question\n<|instruction|>\n\nHow do I create a list of squares of numbers from 1 to 10 in Python?<|eot_id|><|output|>\n\nsquares = [x**2 for x in range(1, 11)]<|eot_id|>\nExample 2: General question or conversational statement\n<|instruction|>\n\nHello, how are you today?<|eot_id|><|output|>\n\nHello! I'm just a language model, so I don't have feelings, but I'm here and ready to help you with any questions you have!<|eot_id|>\nExample 3: Multi-Turn Conversation\n<|instruction|>\n\nHi there!<|eot_id|><|output|>\n\nHello! How can I assist you today?<|eot_id|><|instruction|>\n\nI'm trying to write a Python function to calculate the factorial of a number. Can you help me with that?<|eot_id|><|output|>\n\nSure! Here's a simple Python function to calculate the factorial of a number:\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\nYou can call this function with any non-negative integer to get its factorial.<|eot_id|><|instruction|>\n\nThanks! By the way, do you have any tips for optimizing this function?<|start_header_id|><|output|><|end_header_id|>\n\nYes! To optimize the factorial calculation, you can use memoization to store previously calculated values. Here's an optimized version using a dictionary:\ndef factorial(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n == 0:\n        return 1\n    else:\n        memo[n] = n * factorial(n-1, memo)\n        return memo[n]\nThis reduces the time complexity by avoiding redundant calculations.<|eot_id|><|start_header_id|><|instruction|><|end_header_id|>\n\nThat's great! Also, do you like coding?<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nAs an AI, I don't have personal preferences or feelings, but I do enjoy helping people with coding questions! Is there anything else you need assistance with?<|eot_id|>\n" + '<|eot_id|>' %}{% else %}{% set system  = '<|start_header_id|>' + '<|system|>' + '<|end_header_id|>\n\n'+ "Act like you are an excellent coder and always come up with faithful and trustworthy answers. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.Please remember that if the prompt is related to coding, you should provide a clear, concise, and accurate code example or explanation. Otherwise, if the prompt is a general question or a conversational statement, you should respond in a natural and friendly manner appropriate for casual conversation. Moreover, if you are in multi-turn conversations, you should maintain context and coherence throughout the dialogue, ensuring continuity and relevance in responses. For example:\nExample 1: Coding-related question\n<|start_header_id|><|instruction|><|end_header_id|>\n\nHow do I create a list of squares of numbers from 1 to 10 in Python?<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nsquares = [x**2 for x in range(1, 11)]<|eot_id|>\nExample 2: General question or conversational statement\n<|start_header_id|><|instruction|><|end_header_id|>\n\nHello, how are you today?<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nHello! I'm just a language model, so I don't have feelings, but I'm here and ready to help you with any questions you have!<|eot_id|>\nExample 3: Multi-Turn Conversation\n<|start_header_id|><|instruction|><|end_header_id|>\n\nHi there!<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nHello! How can I assist you today?<|eot_id|><|start_header_id|><|instruction|><|end_header_id|>\n\nI'm trying to write a Python function to calculate the factorial of a number. Can you help me with that?<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nSure! Here's a simple Python function to calculate the factorial of a number:\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\nYou can call this function with any non-negative integer to get its factorial.<|eot_id|><|start_header_id|><|instruction|><|end_header_id|>\n\nThanks! By the way, do you have any tips for optimizing this function?<|start_header_id|><|output|><|end_header_id|>\n\nYes! To optimize the factorial calculation, you can use memoization to store previously calculated values. Here's an optimized version using a dictionary:\ndef factorial(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n == 0:\n        return 1\n    else:\n        memo[n] = n * factorial(n-1, memo)\n        return memo[n]\nThis reduces the time complexity by avoiding redundant calculations.<|eot_id|><|start_header_id|><|instruction|><|end_header_id|>\n\nThat's great! Also, do you like coding?<|eot_id|><|start_header_id|><|output|><|end_header_id|>\n\nAs an AI, I don't have personal preferences or feelings, but I do enjoy helping people with coding questions! Is there anything else you need assistance with?<|eot_id|>\n" + '<|eot_id|>' %}{% endif%}{{bos_token + system}}{% for message in loop_messages %}{% set instruction = '<|start_header_id|>' + '<|instruction|>' + '<|end_header_id|>\n\n'+ message['instruction'] | trim +'<|eot_id|>' %}{% if 'output' not in message.keys()%}{% set output ='' %}{% else %}{% set output = '<|start_header_id|>' + '<|output|>' + '<|end_header_id|>\n\n' + message['output'] | trim + '<|eot_id|>' %}{% endif %}{% if message['input'] ==''%}{% set content = instruction + output %}{% else %}{% set input = '<|start_header_id|>' + '<|input|>' + '<|end_header_id|>\n\n' + message['input'] | trim + '<|eot_id|>' %}{% set content = instruction + input + output %}{% endif%}{{content}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|><|output|><|end_header_id|>\n\n' }}{% endif %}"""

eos_token_id = tokenizer.convert_tokens_to_ids(["<|eot_id|>", "<|end_of_text|>"])
print("eos_token_id: ", eos_token_id)

conversation = []
print("Enter 'reset' to clear the chat history.")
while True:
    human = input("Human:")
    if human.lower() == "reset":
        conversation = []
        print("The chat history has been cleared!")
        continue
  
    conversation.append({"instruction": human, "input": "" })
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05,
        eos_token_id=eos_token_id,
    )
    assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    print("Assistant: ")
    print(assistant)
    conversation[-1]["output"] = assistant