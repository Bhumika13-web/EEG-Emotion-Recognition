import os
import glob

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the corrupted characters exactly as they appear
    original = content
    content = content.replace('🧠', '🧠')
    content = content.replace('✓', '✓')
    content = content.replace('•', '•')
    content = content.replace('', '')
    
    if original != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
