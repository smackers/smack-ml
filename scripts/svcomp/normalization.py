#!usr/bin/env python2.7
import numpy as np
import math

class normalize():
	def __init__(self, fmat,m,n):
		self.fmat = fmat.astype(float)
		self.rows = m
		self.cols = n

	def meanStd(self):
		mean = []; std = []; count = 0

		'''1st column = <filename>'''
		for i in range(1, self.cols-1):
			mu = np.mean(self.fmat[:,i])
			sdev = np.std(self.fmat[:,i])

			mean.append(mu); std.append(sdev)
			if sdev != 0: self.fmat[:,i] = (self.fmat[:,i] - mu)/sdev;
			else: self.fmat[:,i] = self.fmat[:,i] - mu;
		return self.fmat, mean, std
