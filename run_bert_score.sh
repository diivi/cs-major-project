#!/bin/bash

# Usage: ./run_bert_score.sh "<actual_answer>" "<model_answer>"

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 \"<actual_answer>\" \"<model_answer>\""
    exit 1
fi

# Assign arguments to variables
actual_answer="$1"
model_answer="$2"

# Run the bert-score command
bert-score --lang en -r "$actual_answer" -c "$model_answer" -m microsoft/deberta-large-mnli --use_fast_tokenizer
