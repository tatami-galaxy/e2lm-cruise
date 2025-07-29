import json
import glob
import os
from collections import defaultdict

# Configuration
ARCHITECTURE = "dense-500m-arch1"
EXPERIMENT_NAME = ARCHITECTURE
INPUT_DIR = "./"
OUTPUT_FILE = "consolidated_results.json"

# Define your task lists
mmlu_hs_list = [
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics"
]

mmlu_adv_list = [
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics"
]

# Initialize data structure
output = {
    "data": {
        ARCHITECTURE: {
            "model_type": [],
            "experiment": [],
            "iteration": [],
            # Will add benchmark fields dynamically
        }
    }
}

# Create a dictionary to store all results by iteration
results_by_iteration = defaultdict(dict)

# Process all result files
for file_path in glob.glob(f"{INPUT_DIR}/*.json"):
    try:
        # Extract information from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        # Get iteration (step)
        iteration = int(parts[0].replace("step", ""))
        
        # Get task type and name
        if '_adv_' in filename:
            task_name = filename.split('_adv_')[1].split('_results')[0]
            benchmark_prefix = ""
        elif '_hs_' in filename:
            task_name = filename.split('_hs_')[1].split('_results')[0]
            benchmark_prefix = ""
        else:
            continue
        
        # Read the result file
        with open(file_path, 'r') as f:
            result_data = json.load(f)
        
        # Get the accuracy score
        full_task_name = f"mmlu_var5shots_{task_name}"
        accuracy = result_data["results"][full_task_name]["acc,none"]
        
        # Store the result
        benchmark_name = f"{benchmark_prefix}{task_name}"
        print(benchmark_name)
        results_by_iteration[iteration][benchmark_name] = accuracy
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Sort iterations and populate the output structure
sorted_iterations = sorted(results_by_iteration.keys())
for iteration in sorted_iterations:
    # Add metadata
    output["data"][ARCHITECTURE]["model_type"].append("Competition-Proposal")
    output["data"][ARCHITECTURE]["experiment"].append(EXPERIMENT_NAME)
    output["data"][ARCHITECTURE]["iteration"].append(iteration)
    
    # Add benchmark results
    for benchmark, value in results_by_iteration[iteration].items():
        if benchmark not in output["data"][ARCHITECTURE]:
            output["data"][ARCHITECTURE][benchmark] = []
        output["data"][ARCHITECTURE][benchmark].append(value)

# Save the consolidated results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Consolidated results saved to {OUTPUT_FILE}")