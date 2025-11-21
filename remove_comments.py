import os
import re
import sys

def remove_comments_from_file(file_path):
    """Remove all single-line comments from a Python file while preserving docstrings"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        in_triple_quote = False
        triple_quote_char = None
        
        for line in lines:
            stripped = line.lstrip()
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote_char = '"""' if stripped.startswith('"""') else "'''"
                if not in_triple_quote:
                    in_triple_quote = True
                    triple_quote_char = quote_char
                    new_lines.append(line)
                elif triple_quote_char == quote_char:
                    if stripped.count(quote_char) >= 2:
                        in_triple_quote = False
                        triple_quote_char = None
                    new_lines.append(line)
                else:
                    new_lines.append(line)
                continue
            
            if in_triple_quote:
                new_lines.append(line)
                continue
            
            if '#' in line:
                match = re.search(r'^(\s*)#', line)
                if match:
                    continue
                
                parts = line.split('#', 1)
                before_hash = parts[0]
                
                quote_count_single = before_hash.count("'") - before_hash.count("\\'")
                quote_count_double = before_hash.count('"') - before_hash.count('\\"')
                
                if quote_count_single % 2 == 0 and quote_count_double % 2 == 0:
                    cleaned_line = before_hash.rstrip() + '\n'
                    if cleaned_line.strip():
                        new_lines.append(cleaned_line)
                    elif not before_hash.strip():
                        continue
                    else:
                        new_lines.append(cleaned_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory):
    """Process all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != 'remove_comments.py':
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    success_count = 0
    for file_path in python_files:
        print(f"Processing: {file_path}")
        if remove_comments_from_file(file_path):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(python_files)} files processed successfully")

if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    process_directory(directory)
    print("\nAll comments removed successfully!")
