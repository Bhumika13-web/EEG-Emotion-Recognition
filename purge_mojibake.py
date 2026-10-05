import os

def clean_all_nonascii(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original = content
    # Remove all characters > 127
    content = ''.join([c if ord(c) < 128 else '' for c in content])
    
    if original != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Purged mojibake from {filepath}")

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            clean_all_nonascii(os.path.join(root, file))
