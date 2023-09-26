#!/bin/bash

# Remove the directory if it exists
if [ -d "lang-segment-anything" ]; then
  rm -rf lang-segment-anything
fi

# Clone and install lang-segment-anything
git clone https://github.com/luca-medeiros/lang-segment-anything 
cd lang-segment-anything
#pip uninstall torch torchvision 
#pip install torch torchvision 
pip install -e .