#!/bin/bash
# Label Printer Launcher
# Opens the label printer app in your default browser

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start Streamlit
echo "Starting Label Printer..."
streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

# Deactivate virtual environment when done
deactivate
