from implement_idea import run_idea
import json
import os

research_direction = "computer_vision"

# Read the ideas.json file
ideas_file_path = os.path.join("research_directions", research_direction, "ideas.json")

with open(ideas_file_path, 'r') as file:
    ideas = json.load(file)

# Iterate through each idea and run it
for idx, idea in enumerate(ideas, start=1):
    print(f"Running idea {idx}: {idea['Name']}")
    run_idea(idea['Title'], idea['Experiment'], idea['Name'], research_direction)
    print(f"Completed idea {idx}\n")

print("All ideas have been processed.")
