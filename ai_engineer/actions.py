from langchain.schema import AIMessage, HumanMessage, SystemMessage

from ai_engineer.ai import AI
from ai_engineer.extract import chat_to_files_dict



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

    return chat_to_files_dict(messages[-1].content)  # TODO: if extracting fails, ask again!


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
