import glob2, sys

class classify():
	def __init__(self):
		pass

	def readRE(self):
		d = {}; total = 0;
		pathname = '/proj/SMACK/sv-benchmarks/c/'
		setAll = glob2.glob(pathname+'*.set')
		#print setAll
		for i in range(len(setAll)):
			num = self.labelassign(setAll[i])

			if num != None:
				re = [re.rstrip('\n') for re in open(setAll[i])]
				for k in range(len(re)):
					files = glob2.glob(pathname+re[k])
					total = total + len(files)
					for t in range(len(files)):
						if files[t] not in d:
							d[files[t]] = num
		#print total
		return d
		

	def labelassign(self, filename):
		if 'ReachSafety' in filename:
			if 'Arrays' in filename: return 0;
			if 'BitVectors' in filename: return 1;
			if 'ControlFlow' in filename: return 2;
			if 'ECA' in filename: return 3;
			if 'Floats' in filename: return 4;
			if 'Heap' in filename: return 5;
			if 'Loops' in filename: return 6;
			if 'ProductLines' in filename: return 7;
			if 'Recursive' in filename: return 8;
			if 'Sequentialized' in filename: return 9;
			if 'DeviceDriversLinux64' in filename: return 10;
