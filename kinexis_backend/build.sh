#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

# Ensure we're using Python 3.11
python3.11 --version || (echo "Python 3.11 not found!" && exit 1)

# Install dependencies with Python 3.11
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

echo "Build completed successfully!"