import subprocess
import shlex
import pickle
from sklearn import svm
import pickle
import numpy as np
import random

class FeatureGeneration(object):
	def __init__(self):
		pass

	#Purpose: Use tool1 (Yulia) and too2 (Thomas) to generate feature vectors of length 20 and 13 respectively
	def RunTool1(self,input_file):
		output_roles = '../txt/features_tool1.txt'
		output_metrics = '../txt/output_metrics1.txt'
		subprocess.call(['bash','../sh/tool1.sh',input_file,output_roles,output_metrics])

	def RunTool2(self,input_file):
		output_file = '../txt/features_tool2.txt'
		subprocess.call(['bash','../sh/tool2.sh',input_file,output_file])

	#Purpose: Create a dictionary with a type-match key as labels
	def Formatting(self,content, n):
		temp_list = []
		dic = {}

		# ---- list of list containing the filename & features vectors
		if n == 14761:
			for i in range(n):
				listed = content[i].strip().split(' ')
				temp_list.append(listed)
		else:
			for i in range(n):
				listed = content[i].strip().split('\t')
				listed[0] = '../..' + listed[0][13:] #formatting needed to match the FILENAME
				temp_list.append(listed)

		# ---- dic = {filename: [feature-vector]}
		for i in range(n):
			temp = map(float,temp_list[i][1:])
			#if len(temp) == 20:	---- set this flag if the features are not generated of equal length
			dic[temp_list[i][0]] = temp

		return dic

	#Purpose: to combine features from tool 2 with the ones of tool 1
	def MergeFeatures(self,tool1, tool2):
		for filename in tool2:
			if filename in tool1:
				tool1[filename].extend(tool2[filename])
		return tool1
