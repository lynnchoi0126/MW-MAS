import os
import re
import numpy as np
def extract_generated_text(root_dir):
    results = {}
    
    for dir_name in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir_name)
        text_file = os.path.join(dir_path, "text_output.txt")
        
        if os.path.isdir(dir_path) and os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()

            match = re.search(r"Generated \(Watermarked\):\s*(.*?)\n(?:Original|Bit accuracy|Predicted message|P-value)", content, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()
                results[dir_name] = generated_text
            else:
                results[dir_name] = "[Generated text not found]"

    return results

def calculate_bits(text: str, encoding='utf-8') -> int:
    encoded = text.encode(encoding)
    bit_size = len(encoded) * 8  # 1 byte = 8 bits
    return bit_size

# Example usage
root_directory = "paired_output"
generated_outputs = extract_generated_text(root_directory)
n = 0
# Print sample
bit_ls = []
for sample_name, text in generated_outputs.items():
    # print(f"{sample_name}:\n{text}\n{'-'*40}")
    bit_ls.append(calculate_bits(text))
    
np_bit = np.array(bit_ls)
print(f"Total samples: {len(np_bit)}")
print(f"Mean bits: {np.mean(np_bit):.2f}")
print(f"Min bits: {np.min(np_bit):.2f}")

