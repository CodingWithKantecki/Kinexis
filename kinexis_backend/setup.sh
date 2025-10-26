#!/bin/bash

echo "=========================================="
echo "Kinexis Backend Setup"
echo "=========================================="

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is not installed"
    echo "Please install Python 3.11.1 to continue"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo "Note: This project requires Python 3.11.1"

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv venv

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