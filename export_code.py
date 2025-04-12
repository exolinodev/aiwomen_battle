#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
from datetime import datetime

def load_gitignore_patterns(repo_root):
    """Load patterns from .gitignore file."""
    gitignore_path = repo_root / '.gitignore'
    patterns = []
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Convert gitignore pattern to regex pattern
                pattern = line.replace('.', r'\.').replace('*', '.*').replace('?', '.')
                
                # Handle directory indicators
                if pattern.endswith('/'):
                    pattern = pattern[:-1] + '(?:/|$)'
                
                patterns.append(re.compile(pattern))
    
    return patterns

def should_ignore(path, patterns):
    """Check if a path should be ignored based on gitignore patterns."""
    str_path = str(path)
    
    for pattern in patterns:
        if pattern.search(str_path):
            return True
    
    # Common files and directories to ignore even if not in gitignore
    common_ignores = [
        r'\.git/',
        r'__pycache__/',
        r'\.pytest_cache/',
        r'\.mypy_cache/',
        r'\.coverage',
        r'\.tox/',
        r'\.idea/',
        r'\.vscode/',
        r'\.DS_Store'
    ]
    
    for pattern in common_ignores:
        if re.search(pattern, str_path):
            return True
    
    return False

def is_text_file(file_path):
    """Check if a file is a text file based on extension."""
    text_extensions = {
        '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.html', '.css', '.js',
        '.sh', '.bat', '.cfg', '.ini', '.conf', '.toml', '.rst', '.csv',
        '.gitignore', '.env.example', '.dockerignore', '.editorconfig',
        '.md', '.rst', '.ipynb'
    }
    
    return file_path.suffix.lower() in text_extensions

def export_code(repo_root, output_file):
    """Export all code files to a single text file."""
    patterns = load_gitignore_patterns(repo_root)
    
    with open(output_file, 'w') as f:
        f.write(f"# Code Export from {repo_root}\n")
        f.write(f"# Generated on {datetime.now()}\n\n")
        
        for path in sorted(repo_root.glob('**/*')):
            if path.is_file() and is_text_file(path) and not should_ignore(path, patterns):
                relative_path = path.relative_to(repo_root)
                
                f.write(f"\n{'='*80}\n")
                f.write(f"# FILE: {relative_path}\n")
                f.write(f"{'='*80}\n\n")
                
                try:
                    with open(path, 'r', encoding='utf-8') as code_file:
                        content = code_file.read()
                        f.write(content)
                        if not content.endswith('\n'):
                            f.write('\n')
                except Exception as e:
                    f.write(f"# ERROR reading file: {e}\n")

if __name__ == "__main__":
    # Use the current directory as repo root if not specified
    repo_root = Path(os.getcwd())
    output_file = repo_root / "code_export.txt"
    
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    
    print(f"Exporting code from {repo_root} to {output_file}")
    export_code(repo_root, output_file)
    print(f"Export completed: {output_file}")