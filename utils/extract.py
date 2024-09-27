def extract_json_between_markers(llm_output):
    json_start_marker = "```json"
    json_end_marker = "```"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)  # Move past the marker
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None  # JSON markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    json_string = llm_output[start_index:end_index].strip()
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None  # Invalid JSON format