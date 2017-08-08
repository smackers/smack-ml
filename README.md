# SMACK_ML

/scripts/src/Features.py: A python class that generates, merges all the feature vectors from both tools and writes in a file ../txt/FinalFeatures.txt as a dictionary data structure. The pre-compiled features at the .txt file can be directly used for testing.

/scripts/src/labels.py: A python class that parses the .xml documents for a given flag, extracts all possible labels based on the cputime and finally computes the minimum cputime for each file. The result is a dictionary data structure with {filename: minimum cputime}

The data files can be found under another repository /sv-benchmarks.
