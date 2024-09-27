import os
import yaml
from tqdm import tqdm   
from pathlib import Path

from utils.completion import post
from utils.extract import extract_json_between_markers

DIRECTIONS_PATH = Path("research_directions")


IDEA_PROMPT = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, required=False, help="Specific research direction to generate ideas for")
    parser.add_argument("--num_ideas", type=int, default=5)
    args = parser.parse_args()

    if args.direction:
        assert args.direction in os.listdir(DIRECTIONS_PATH), f"Direction {args.direction} not found in {DIRECTIONS_PATH}"
        direction_paths = [DIRECTIONS_PATH / args.direction]
    else:
        direction_paths = [d for d in DIRECTIONS_PATH.iterdir() if d.is_dir()]
        
    
    for direction_path in direction_paths:        
        with open(direction_path / "prompt.yaml", "r") as f: prompt = yaml.safe_load(f) or []
        with open(direction_path / "utils.py", "r") as f: code = f.read() or []
        with open(direction_path / "ideas.yaml", "r") as f: prev_ideas = yaml.safe_load(f) or []
    
        for _ in tqdm(range(args.num_ideas)):
            idea_prompt = IDEA_PROMPT.format(
                task_description=prompt["task_description"],
                code=code,
                prev_ideas_string=yaml.dump(prev_ideas),
            )
            msg = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": idea_prompt}
            ]
            
            idea = post("openai/o1-preview-2024-09-12", msg)

            # parse the ide
            idea_json = extract_json_between_markers(idea)
            assert idea_json, "No JSON found in the idea"
            
            prev_ideas.append(idea_json)
        
            
            # create the folder, save the idea (maybe _json is enough, otherwise we could ask for a more thorough writeup)
            
            # maybe do some reflection here, perhaps o1 does that well enough already
        
        
        
        
        
        
        
            
        
            
            
            
            
            
            
            
            
        
        
    
    
    









# import os
# import dotenv
# import requests
# from typing import List, Optional

# dotenv.load_dotenv()

# API_KEY = os.getenv("OPENROUTER_API_KEY")
# API_ENDPOINT = os.getenv("OPENROUTER_API_ENDPOINT")

# def post(self, chat_history: Optional[List]=None):
#     response = requests.post(
#         API_ENDPOINT,
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {API_KEY}",
#         },
#         json={
#             "model": "openai/o1-preview-2024-09-12",
#             "messages": chat_history or [],
#             "temperature": self.temperature,
#             "top_p": self.top_p
#         },
#     )
#     response_json = response.json()

#     print(self.messages)
#     print(chat_history)
#     print(response_json)
#     return response_json["choices"][0]["message"]["content"]