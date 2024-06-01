import os

# Directory containing your data files
code_dir = "data/bigcode_the-stack-smol/content_test"
response_dir = 'data/bigcode_the-stack-smol/response'
output_dir = 'data/bigcode_the-stack-smol/merged'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

total_pairs = 250  
invalid_file = [84, 99, 113, 124, 157, 164, 165, 166, 174, 178, 179, 203, 209, 223]

for i in range(1, total_pairs + 1):
    if i in invalid_file: continue
    
    code_file = os.path.join(code_dir, f'content_{i}.txt')
    response_file = os.path.join(response_dir, f'response_{i}.txt')
    output_file = os.path.join(output_dir, f'merge_{i}.txt')

    with open(code_file, 'r', encoding='utf-8') as code, open(response_file, 'r', encoding='utf-8') as response, open(output_file, 'w', encoding='utf-8') as outfile:
        # Read the contents from the files

        ## think of system_prompt
        system_prompt = "You are an expert in programming, particularly in Python. Your task is to explain complex code snippets succinctly. Focus on providing clear, concise instructions that describe what the code does and how it works, suitable for use in training a code generation model."
        code_content = code.read().strip()
        response_content = response.read().strip()

        # Format the merged content
        merged_content = f"<s>[INST] <<SYS>>\n{{ {system_prompt} }}\n<</SYS>>\n{{ {response_content} }} [/INST] {{ {code_content} }}\n</s>"
        # Write the formatted content to the new file
        outfile.write(merged_content)

print("Merging completed successfully.")