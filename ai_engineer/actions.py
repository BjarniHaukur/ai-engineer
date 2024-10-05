import os
from pathlib import Path
from langchain.schema import HumanMessage, SystemMessage

from ai_engineer.ai import AI
from ai_engineer.filesdict import FilesDict

OUTPUT_DIR = Path(".." if os.path.exists("../.git") else ".") / "outputs"

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
    return FilesDict.from_response(messages[-1].content, root_path=OUTPUT_DIR)

def generate_improvement(ai:AI, error:str, files_dict:FilesDict)->FilesDict:
    messages = [
        SystemMessage(content=prompts["roadmap"] + prompts["improve"].replace("FILE_FORMAT", prompts["file_format_diff"]) + "\nUseful to know:\n" + prompts["philosophy"]),
        HumanMessage(content=f"Error: {error}\n\nFiles:\n{files_dict.to_context(enumerate_lines=True)}"),
    ]

    messages = ai.next(messages)
    return FilesDict.from_response(messages[-1].content, root_path=OUTPUT_DIR)
