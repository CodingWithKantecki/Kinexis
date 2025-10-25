#!/bin/bash

echo "=========================================="
echo "Kinexis Backend Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the backend server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run server: python app.py"
echo ""
echo "The server will be available at:"
echo "- HTTP: http://localhost:5000"
echo "- WebSocket: ws://localhost:5000"
echo ""
echo "To test with webcam:"
echo "python test_client.py"
echo ""