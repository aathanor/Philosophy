#!/bin/bash
# Setup script for Zettelkasten Label Printer

echo "🏷️  Zettelkasten Label Printer Setup"
echo "===================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml to configure your printer and folders"
echo "2. Place markdown exports in your configured folders"
echo "3. Run the app with: streamlit run app.py"
echo ""
echo "For help, see README.md"
