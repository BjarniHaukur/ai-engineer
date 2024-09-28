import json

def extract(text:str, start_marker:str):
    start_marker, end_marker = "```" + start_marker, "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    
    if start_index == -1 or end_index == -1: raise ValueError("Start or end marker not found")
    
    return text[start_index:end_index].strip()
    
def extract_json(text:str):
    try: return json.loads(extract(text, "json"))
    except json.JSONDecodeError: raise ValueError("Invalid JSON")
    