#!/bin/bash

# Set the script to exit on any error
set -e

# Specify the paths to your Python files
PYTHON_SCRIPT="data_construct.py"
STREAMLIT_APP="app.py"

# Run the first Python script with python3.12
echo "Running $PYTHON_SCRIPT with python3.12..."
python3.12 $PYTHON_SCRIPT   # Run in the background

# Run the Streamlit app
echo "Running Streamlit app: $STREAMLIT_APP..."
streamlit run $STREAMLIT_APP

# Indicate completion
echo "Both scripts have been executed!"
