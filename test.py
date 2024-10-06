from ai_engineer.ai import AI
from ai_engineer.actions import generate_code, generate_bash, generate_improvement
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv
from ai_engineer.diff import extract_all_diffs
from ai_engineer.utils import strip_ansi_codes


# ai_big = AI(model_name="openai/o1-preview-2024-09-12")
# ai_big = AI(model_name="openai/o1-mini-2024-09-12")
ai_small = AI(model_name="gpt-4o", stdout=True, temperature=0.5)
env = ExecutionEnv("outputs", stderr=True)

# files_dict = generate_code(ai_small, "Make a simple snake game")
files_dict = FilesDict.from_folder("outputs")
command = generate_bash(ai_small, files_dict)
# command ="""uv pip install -r requirements.txt
# uv run src/main.py"""

print(command)


while True:
    stdout, stderr, error_code = env.run(command)

    print("Want to change anything?")
    do_fix = input("description/n: ")
    if do_fix == "n":
        break
    
    response = generate_improvement(ai_small, do_fix, files_dict)
    diffs = extract_all_diffs(response)
    for diff in diffs:
        files_dict = diff.apply(files_dict)
