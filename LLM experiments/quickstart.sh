#!/bin/bash
# Quick start script for SEP Alpaca preparation

set -e  # Exit on error

echo "================================================"
echo "SEP Alpaca LLM Preparation - Quick Start"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements_minimal.txt -q

echo "✓ Dependencies installed"
echo ""

# Run the script
echo "================================================"
echo "Starting SEP download (first 10 articles for testing)"
echo "================================================"
echo ""

python3 sep_alpaca_prep.py --max-articles 10 --output-dir ./sep_test

echo ""
echo "================================================"
echo "✓ Test run complete!"
echo "================================================"
echo ""
echo "Results saved to: ./sep_test/"
echo ""
echo "To download all articles, run:"
echo "  source venv/bin/activate"
echo "  python3 sep_alpaca_prep.py --max-articles 500"
echo ""
echo "View the Alpaca dataset:"
echo "  cat sep_test/alpaca_datasets/alpaca_philosophy.json | head -100"
echo ""
