#!/bin/bash
# render-build.sh - Build script for Render

echo "=== Starting Render build process ==="

# Exit on error
set -e

# Install system dependencies for Tesseract OCR
echo "Installing Tesseract OCR..."
apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation
echo "Tesseract version:"
tesseract --version

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating temporary directories..."
mkdir -p /tmp/uploads /tmp/plates_detected /tmp/matplotlib /tmp/cache /tmp/config /tmp/data /tmp/torch

echo "=== Build complete ==="