import re

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from ai_engineer.ai import AI


def chat_to_files_dict(chat:str)->dict[str, str]:
    # Regex to match file paths and associated code blocks
    regex = r"(\S+)\n\s*```[^\n]*\n(.+?)```"
    matches = re.finditer(regex, chat, re.DOTALL)

    files_dict = {}
    for match in matches:
        # Clean and standardize the file path
        path = re.sub(r'[\:<>"|?*]', "", match.group(1))
        path = re.sub(r"^\[(.*)\]$", r"\1", path)
        path = re.sub(r"^`(.*)`$", r"\1", path)
        path = re.sub(r"[\]\:]$", "", path)

        # Extract and clean the code content
        content = match.group(2)

        files_dict[path.strip()] = content.strip()

    return files_dict


def generate_code(ai:AI, prompt:str)->dict[str, str]:
    roadmap = open("ai_engineer/prompts/roadmap").read()
    generate = open("ai_engineer/prompts/generate").read()
    file_format = open("ai_engineer/prompts/file_format").read()
    philosophy = open("ai_engineer/prompts/philosophy").read()

    messages = [
        SystemMessage(content=roadmap + generate.replace("FILE_FORMAT", file_format) + "\nUseful to know:\n" + philosophy),
        HumanMessage(content=prompt),
    ]

    messages = ai.next(messages)

    return chat_to_files_dict(messages[-1].content)


def fix_code():
    pass

def improve_code():
    pass

def run_code():
    pass


# def fix_code(ai:AI, prompt:str)->dict[str, str]:
#     roadmap = open("ai_engineer/prompts/roadmap").read()
#     improve = open("ai_engineer/prompts/improve").read()
#     file_format_diff = open("ai_engineer/prompts/file_format_diff").read()
#     philosophy = open("ai_engineer/prompts/philosophy").read()

#     messages = [
#         SystemMessage(content=roadmap + improve.replace("FILE_FORMAT", file_format_diff) + "\nUseful to know:\n" + philosophy),
#         HumanMessage(content=prompt),
#     ]

#     messages = ai.next(messages)

#     return chat_to_files_dict(messages[-1].content)
