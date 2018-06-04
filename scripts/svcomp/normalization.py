import numpy as np
import math

class normalize():
	def __init__(self, fmat,m,n):
		self.fmat = fmat
		self.rows = m
		self.cols = n

	def meanNorm(self):
		mean = []; std = []; count = 0
		#print self.fmat
		for i in range(self.cols-1):
			mu = np.mean(self.fmat[:,i])
			sdev = np.std(self.fmat[:,i])
			#print mu, sdev
			mean.append(mu); std.append(sdev)
			if sdev != 0: self.fmat[:,i] = (self.fmat[:,i] - mu)/sdev;
			else: self.fmat[:,i] = self.fmat[:,i] - mu;
		return self.fmat, mean, std
		print self.fmat[:,5]
		
			


