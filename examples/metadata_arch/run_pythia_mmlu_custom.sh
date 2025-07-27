#!/bin/bash

OUTPUT_DIR="./results/"
BATCH_SIZE="8"

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
#echo "STEPS = $STEPS"
#echo "REPOSITORY_NAME = $REPOSITORY_NAME"
#echo "EXTRA_VALUES = $EXTRA_VALUES"


for (( i=1; i<=143; i++ ))
do
	lm_eval --model hf --model_args pretrained=$MODEL,revision=step$(( i * 1000 )),dtype="float" --tasks $TASK --device cuda:$GPU --batch_size $BATCH_SIZE --output_path $OUTPUT_DIR
done

