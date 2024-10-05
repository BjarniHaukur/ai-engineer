from ai_engineer.ai import AI
from ai_engineer.actions import generate_code, generate_bash
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv


# ai = AI(model_name="openai/o1-preview-2024-09-12")
ai = AI(model_name="gpt-4o", stdout=True)
env = ExecutionEnv("outputs", stderr=True)

# files_dict = generate_code(ai, "Snake game where you can control the snake with the arrow keys")
files_dict = FilesDict.from_folder("outputs")
command = generate_bash(ai, files_dict)
env.run(command)