#!/bin/bash
# Simple script to run StatAgent demo
# Run this with: bash run_demo.sh

echo "Setting up StatAgent demo..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q numpy scipy matplotlib pandas

# Install package
echo "Installing StatAgent..."
pip install -q -e .

# Run demo
echo ""
echo "Running demo..."
echo ""
python demo.py

echo ""
echo "Demo complete! Check the figures/ directory for visualizations."

