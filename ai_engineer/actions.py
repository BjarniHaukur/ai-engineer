from pathlib import Path
from langchain.schema import HumanMessage, SystemMessage

from ai_engineer.ai import AI
from ai_engineer.file import FileDict
prompts = {
    path.stem: path.read_text()  # fails if 'path' is not a file, expected behaviour
    for path in Path("ai_engineer/prompts").iterdir()
}


def generate_code(ai:AI, prompt:str)->FileDict:
    messages = [
        SystemMessage(content=prompts["roadmap"] + prompts["generate"].replace("FILE_FORMAT", prompts["file_format"]) + "\nUseful to know:\n" + prompts["philosophy"]),
        HumanMessage(content=prompt),
    ]
    messages = ai.next(messages)
    return FileDict.from_response(messages[-1].content)
