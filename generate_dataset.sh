#!/bin/bash

# Directory containing dataset files
dataset_dir="data/mp3d"

# Command to run creating_pointnav_dataset.py
script_path="scripts/creating_pointnav_dataset.py"
base_command="python $script_path -d mp3d -s"

# Output log file
log_file="script_log.txt"
if [ -e "$log_file" ]; then
    rm "$log_file"
    echo "Log file $log_file has been deleted."
fi

# Flag to check if Ctrl+C is pressed
skip_current_command=false

# Function to handle Ctrl+C
function handle_ctrl_c {
    echo "Custom shortcut Ctrl+C pressed. Skipping the current command."
    skip_current_command=true
}

# Set up Ctrl+C handler
trap handle_ctrl_c INT

# Iterate over dataset files and run the script
for dataset_file in "$dataset_dir"/*; do

    read -t 5 -n 1 key
    if [ "$key" == "k" ]; then
        echo "Exit Script by pressing k"
        break  # Skip empty key presses K
    fi

    # Check if the file exists and has the correct extension
    if [ -e "$dataset_file" ]; then
        # Extract the file name without the directory and extension
        file_name="$(basename "$dataset_file")"
        
        # Build the full command
        full_command="$base_command $file_name"
        
        # Execute the command
        echo "Running: $full_command"
        $full_command
        
        # Add any additional logic or error handling if needed
    fi

    # Check if Ctrl+C is pressed and exit the loop
    if [ "$skip_current_command" = true ]; then
        echo "Skipping current command for file with error: $dataset_file" >> "$log_file"
        skip_current_command=false
        continue
    else
        echo "Success creating dataset for file: $dataset_file" >> "$log_file"
    fi
done