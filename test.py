from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ai_engineer.ai import AI
from ai_engineer.actions import generate_code


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ai = AI(model_name="gpt-4o")

files_dict = generate_code(ai, "")

for file_path, file_content in files_dict.items():
    path = OUTPUT_DIR / file_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(file_content)
