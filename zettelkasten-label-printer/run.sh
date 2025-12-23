#!/bin/bash
# Quick run script for Zettelkasten Label Printer

echo "🏷️  Starting Zettelkasten Label Printer..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed."
    echo "Run ./setup.sh first to install dependencies."
    exit 1
fi

# Run the app
streamlit run app.py
