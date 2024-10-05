from ai_engineer.ai import AI
from ai_engineer.actions import generate_code, generate_improvement
from ai_engineer.filesdict import FilesDict


# ai = AI(model_name="openai/o1-preview-2024-09-12")
ai = AI(model_name="gpt-4o", stdout=True)

files_dict = generate_code(ai, "Snake game where you can control the snake with the arrow keys")