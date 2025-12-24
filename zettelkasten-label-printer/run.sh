#!/bin/bash
# Quick run script for Zettelkasten Label Printer

echo "🏷️  Starting Zettelkasten Label Printer..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found."
    echo "Run ./setup.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed in virtual environment."
    echo "Run ./setup.sh first to install dependencies."
    deactivate
    exit 1
fi

echo "Starting Streamlit app..."
echo ""

# Run the app
streamlit run app.py

# Deactivate when done (only reached if streamlit exits)
deactivate
