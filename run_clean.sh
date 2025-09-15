#!/bin/bash

# Clean and rebuild the code
make clean
make

# Clear dump files
rm -rf dumps/*

echo "Code rebuilt, and dumps directory cleaned."

