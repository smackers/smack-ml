import subprocess, shlex
from sklearn import svm
import pickle, random, glob2
import numpy as np

class FeatureGeneration(object):
	def __init__(self):
		pass

	#Purpose: Use tool1 (Yulia-clang) to generate feature vectors of length 20
	def RunTool1(self,input_file,output_roles,output_metrics):
		subprocess.call(['bash','../sh/tool1.sh',input_file,output_roles,output_metrics])

	#Purpose: Use tool1 (Thomas-sloopy) to generate feature vectors of length 13
	def RunTool2(self,input_file,output_file):
		subprocess.call(['bash','../sh/tool2.sh',input_file,output_file])

	def Filesize(self,input_file):
			subproces.call(input_file)

	#Extracting file based on FileExtension
	def ExtractFiles(pathname,extension):
		filetype = glob2.glob(pathname + '/*.' + extension)
		return filetype

	#Purpose: Create a dictionary with a type-match key as labels
	def Formatting(self,content, n, SplitParameter):
		temp_list = []
		dic = {}

		'''
		# ---- list of list containing the filename & features vectors
		if n == 14761:
			for i in range(n):
				listed = content[i].strip().split(SplitParameter)
				temp_list.append(listed)
		else:
			for i in range(n):
				listed = content[i].strip().split('\t')
				listed[0] = '../..' + listed[0][13:] #formatting needed to match the FILENAME
				temp_list.append(listed)
		'''

		for i in range(n):
			listed = content[i].strip().split(SplitParameter)
			temp = map(float,listed[1:])

			#if the features vectors are not of equal length (precaution for feature matrix)
			if SplitParameter == ' ' and len(temp) == 20:
				dic[listed[0]] = temp
			elif SplitParameter == '\t' and len(temp) == 13:
				dic[listed[0]] = temp
		return dic

	#Purpose: to combine features from tool 2 with the ones of tool 1
	def MergeFeatures(self,tool1, tool2):
		for filename in tool2:
			if filename in tool1:
				tool1[filename].extend(tool2[filename])
		return tool1
