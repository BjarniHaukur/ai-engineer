from pathlib import Path

from ai_engineer.ai import AI
from ai_engineer.actions import generate_code


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ai = AI(model_name="openai/o1-preview-2024-09-12")
ai = AI(model_name="gpt-4o")

files_dict = generate_code(ai, """Make an autonomous coding agent that can write code, run it, catch errors, fix them and repeat.
    Assume that OPENAI_API_KEY is already set in the environment and use langchain_openai.ChatOpenAI.
    Make the main file use argparse so that the """)

files_dict.to_file(OUTPUT_DIR)