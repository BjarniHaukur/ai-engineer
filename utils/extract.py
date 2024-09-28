import json

def extract(text:str, start_marker:str)->str|None:
    start_marker, end_marker = "```" + start_marker, "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index+len(start_marker))
    
    if start_index == -1 or end_index == -1:
        return None
    
    return text[start_index+len(start_marker):end_index].strip()
    
def extract_json(text:str)->dict|None:
    try:
        extracted = extract(text, "json")
        return json.loads(extracted)
    except json.JSONDecodeError or AttributeError:
        return None