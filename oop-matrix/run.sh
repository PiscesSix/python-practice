#!/bin/bash

# Run flake8
echo "Running flake8 for $1"
flake8 "$1"

# Run mypy
echo "Running mypy for $1"
mypy "$1"
