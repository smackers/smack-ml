#!/usr/bin/env python2.7
import pickle
import numpy as np


'''Goals:
1. make lists
2. remove junk from file name and rename with path = '/proj/SMACK/sv-benchmarks/c'
3. Normalize the features
4. Convert into dictionary
5. Store into a file <path_new = path + '/output/file_name'>
'''
class processing():
    def __init__(self, fname, path, k):
        self.fname = fname
        self.path = path
        self.k = k

    #Goal 1, 2
    def make_lists(self,splitParameter):
        content = []
        with open(self.fname,'r') as f:
			content = f.readlines()
        content = [x.strip().split(splitParameter) for x in content]

        for i in range(len(content)):
            content[i][0] = self.path + content[i][0][self.k:]
            content[i][1:] = map(float,content[i][1:])

        print len(content), len(content[0])
        return content
