import sys, os, pickle
import numpy as np


'''Goals:
1. make lists
2. remove junk from file name and rename with path = '/proj/SMACK/sv-benchmarks/c'
3. Normalize the features
4. Convert into dictionary
5. Store into a file <path_new = path + '/output/file_name'>
'''
class processing():
    def __init__(self, fname):
        self.fname = fname
        path = '/proj/SMACK/smack-ml/scripts/merge/'

    def make_lists(self):
        with open(fname,'r') as f:
			content1 = f.readlines()

		content_tool1 = [x.strip() for x in content1]
