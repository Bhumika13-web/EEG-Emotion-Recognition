import os

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    content = content.replace('\ufffd', '')
    content = content.replace('\uFFFD', '')
    content = content.replace('', '')
    content = content.replace('\ufffd', ' ')
    content = content.replace('\uFFFD', ' ')
    
    if original != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned {filepath}")

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            clean_file(os.path.join(root, file))
