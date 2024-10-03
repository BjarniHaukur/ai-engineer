from pathlib import Path
from langchain.schema import HumanMessage, SystemMessage

from ai_engineer.ai import AI
from ai_engineer.filesdict import FilesDict
from ai_engineer.execution_env import ExecutionEnv
from ai_engineer.constants import RUN_COMMAND

prompts = {
    path.stem: path.read_text()  # fails if 'path' is not a file, expected behaviour
    for path in Path("ai_engineer/prompts").iterdir()
}


def generate_code(ai:AI, prompt:str)->FilesDict:
    messages = [
        SystemMessage(content=prompts["roadmap"] + prompts["generate"].replace("FILE_FORMAT", prompts["file_format"]) + "\nUseful to know:\n" + prompts["philosophy"]),
        HumanMessage(content=prompt),
    ]
    messages = ai.next(messages)
    return FilesDict.from_response(messages[-1].content)

def generate_exec_command(ai:AI, files_dict:FilesDict)->FilesDict:

    messages = [
        SystemMessage(content= prompts["py_exec_command"]),
        HumanMessage(
            content= 
                    "\nInformation about the codebase:\n\n"
                     + files_dict.to_context()
        ),
    ]
    messages = ai.next(messages)
    response = messages[-1].content.strip()
    
    import re
    regex = r"```\S*\n(.+?)```"
    matches = re.finditer(regex, response, re.DOTALL)
    files_dict[RUN_COMMAND] = "\n".join(match.group(1) for match in matches)
    
    return files_dict

def run_code(ai:AI, env:ExecutionEnv, files_dict:FilesDict):
    files_dict = generate_exec_command(ai, files_dict)
    env.upload(files_dict)
    assert RUN_COMMAND in files_dict, "Entrypoint not found"
    stdout, stderr, returncode = env.run(files_dict[RUN_COMMAND])
    return files_dict, stdout, stderr, returncode

    
