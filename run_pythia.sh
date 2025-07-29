#!/bin/bash

# Base model and output settings
MODEL="EleutherAI/pythia-70m-deduped"
OUTPUT_DIR="results"

# Define MMLU task lists
mmlu_hs_list=(
    "high_school_biology" "high_school_chemistry" "high_school_computer_science"  "high_school_mathematics" "high_school_physics"
)

mmlu_adv_list=(
    "college_biology" "college_chemistry" "college_computer_science" "college_mathematics" "college_physics"
)

# Combine all tasks for full evaluation (optional)
mmlu_all_list=("${mmlu_hs_list[@]}" "${mmlu_adv_list[@]}")

# Loop through revisions (step3000 to step15000, interval 1000)
for (( step=18000; step<=143000; step+=1000 )); do
    revision="step${step}"
    echo "===== Evaluating revision: $revision ====="

    # Loop through high school + elementary tasks
    for task in "${mmlu_hs_list[@]}"; do
        echo "[HS/Elementary] Running task: $task"
        lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL,revision=$revision" \
            --tasks "mmlu_var5shots_${task}" \
            --batch_size auto \
            --output_path "$OUTPUT_DIR/${revision}_hs_${task}_results.json"
    done

    # Loop through college + professional tasks
    for task in "${mmlu_adv_list[@]}"; do
        echo "[College/Professional] Running task: $task"
        lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL,revision=$revision" \
            --tasks "mmlu_var5shots_${task}" \
            --batch_size auto \
            --output_path "$OUTPUT_DIR/${revision}_adv_${task}_results.json"
    done

    echo "===== Completed revision: $revision ====="
    echo ""
done

echo "All evaluations completed. Results saved in $OUTPUT_DIR/"

python merge.py

python plot.py
