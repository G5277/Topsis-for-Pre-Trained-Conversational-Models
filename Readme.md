Topsis Implementation and Python Package
This repository contains the implementation of the Technique for Order Preference by Similarity to Ideal Solution (TOPSIS). It includes two components:

Command-Line Python Program
Python Package Published on PyPI
1. Command-Line Python Program
This program calculates the TOPSIS scores and ranks for a given dataset.

How It Works:
Input File:
The input file is a CSV file with three or more columns:

The first column contains the names of the objects/variables (e.g., M1, M2, M3, etc.).
The remaining columns contain numeric data.
Output File:
The output file includes all columns from the input file with two additional columns:

Topsis Score
Rank
Features:
Handles exceptions for:
Incorrect input format.
Missing files.
Non-numeric values.
Ensures weights and impacts are correctly formatted.
Validates the consistency of input dimensions (number of weights, impacts, and columns).
Usage:
Run the program through the command line:

python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>
Example:

python Topsis-Gazal-102217174.py data.csv "1,1,1,2" "+,+,-,+" result.csv
