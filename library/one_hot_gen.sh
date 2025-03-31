#!/bin/bash
#SBATCH --job-name=one_hot_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G  # Request 16 GB (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Check if grammar_discovery.py and make_expressions.py exist
if [ ! -f "grammar_discovery.py" ]; then
    echo "Error: grammar_discovery.py not found!"
    exit 1
fi

if [ ! -f "make_expressions.py" ]; then
    echo "Error: make_expressions.py not found!"
    exit 1
fi

# Set default values
CSV_PATH="/cluster/work/math/camlab-data/symbolic-GED/datasets/test-fix-Derivative-1000.csv"
COLUMN_NAME="manufactured_solution_u"
OUTPUT_FILE="manufactured_solution_u-91len-1000.h5"
MAX_LEN=91

# Check if a different CSV path was provided as an argument
if [ "$1" != "" ]; then
    CSV_PATH="$1"
fi

# Check if a different column name was provided as an argument
if [ "$2" != "" ]; then
    COLUMN_NAME="$2"
fi

# Check if a different output filename was provided as an argument
if [ "$3" != "" ]; then
    OUTPUT_FILE="$3"
fi

# Check if a different max_len was provided
if [ "$4" != "" ]; then
    MAX_LEN="$4"
fi

echo "Using CSV file: $CSV_PATH"
echo "Using column: $COLUMN_NAME"
echo "Output will be saved to: $OUTPUT_FILE"
echo "Max sequence length: $MAX_LEN"

# Run the make_expressions.py script with the parameters
python make_expressions.py "$CSV_PATH" "$COLUMN_NAME" "$OUTPUT_FILE" "$MAX_LEN"

# Check if the output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo "Success! Output file $OUTPUT_FILE created."
else
    echo "Error: Output file was not created."
    exit 1
fi