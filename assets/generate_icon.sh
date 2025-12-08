#!/bin/bash

# Script to generate Icon.ico from Icon.png
# Requires ImageMagick to be installed

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set input and output files
INPUT_PNG="$SCRIPT_DIR/Icon.png"
OUTPUT_ICO="$SCRIPT_DIR/Icon.ico"

# Check if Icon.png exists
if [ ! -f "$INPUT_PNG" ]; then
    echo "Error: Icon.png not found in $SCRIPT_DIR"
    exit 1
fi

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install imagemagick"
    echo "  macOS: brew install imagemagick"
    exit 1
fi

# Generate the .ico file with multiple sizes
echo "Generating Icon.ico from Icon.png..."
convert "$INPUT_PNG" -define icon:auto-resize=256,128,64,48,32,16 "$OUTPUT_ICO"

if [ $? -eq 0 ]; then
    echo "Successfully generated Icon.ico"
    ls -lh "$OUTPUT_ICO"
else
    echo "Error: Failed to generate Icon.ico"
    exit 1
fi
