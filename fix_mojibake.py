import os

def fix_mojibake(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    # Replace known mojibake manually
    content = content.replace('\xf0\u0178\xa7\xa0', '🧠')
    content = content.replace('\xf0\u0178\u201d\u0161', '📈')
    content = content.replace('\xf0\u0178\u201c\u0161', '📈')
    content = content.replace('\xf0\u0178\u02dc\u201c', '📈')
    # Let's just catch the st.set_page_config ones:
    content = content.replace('page_icon="\xf0\u0178\xa7\xa0"', 'page_icon="🧠"')
    
    # Also fix st.success
    content = content.replace('\xe2\u20ac\u0153', '"')
    content = content.replace('\xe2\u20ac\x9d', '"')
    
    if original != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed mojibake in {filepath}")

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            fix_mojibake(os.path.join(root, file))
