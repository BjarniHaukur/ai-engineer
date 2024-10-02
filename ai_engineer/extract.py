import re
import json


def extract(text:str, start_marker:str)->str:
    start_marker, end_marker = "```" + start_marker, "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index+len(start_marker))
    
    if start_index == -1 or end_index == -1:
        raise ValueError("Start or end marker not found in text")
    
    return text[start_index+len(start_marker):end_index].strip()
    
def extract_json(text:str)->dict:
    extracted = extract(text, "json")
    return json.loads(extracted)


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