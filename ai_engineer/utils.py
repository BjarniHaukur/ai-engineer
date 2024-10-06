import re
from pathlib import Path

def is_inconsequential(file_path: str | Path) -> bool:
    path = Path(file_path)
    return (path.name.startswith('.') or
            path.suffix in ['.pyc', '.pyo'] or
            '__pycache__' in path.parts or
            any(pattern in path.name for pattern in ['~', '.tmp', '.bak', '.swp', '.DS_Store']) or
            (path.is_dir() and not any(path.iterdir())))  # empty directory

def strip_ansi_codes(text):
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)