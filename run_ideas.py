from implement_idea import run_idea
import json
import os
import multiprocessing

def process_idea(idea, idx, research_direction):
    print(f"Running idea {idx}: {idea['Name']}")
    run_idea(idea['Title'], f"{idea['Experiment']} {idea['Thought']}", idea['Name'], research_direction)
    print(f"Completed idea {idx}\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide a research direction as a command line argument.")
        sys.exit(1)
    
    research_direction = sys.argv[1]

    # Read the ideas.json file
    ideas_file_path = os.path.join("research_directions", research_direction, "ideas.json")

    with open(ideas_file_path, 'r') as file:
        ideas = json.load(file)

    # Create a pool of worker processes
    num_processes = 5
    pool = multiprocessing.Pool(processes=num_processes)

    # Use pool.starmap to parallelize the processing of ideas
    pool.starmap(process_idea, [(idea, idx, research_direction) for idx, idea in enumerate(ideas, start=1)])

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All ideas have been processed.")
