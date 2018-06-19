import glob2, sys, pickle

'''Goal: Create labels based on SVCOMP category'''

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


if __name__ == '__main__':
	d = {};
	path = '/proj/SMACK/sv-benchmarks/c/' #same path passed tp features.py

    #*.set files contain regex for files for each category
	setAll = glob2.glob(path + '*.set')

	for i in range(len(setAll)):
		num = self.labelassign(setAll[i])

		if num != None:
			re = [re.rstrip('\n') for re in open(setAll[i])]
			for k in range(len(re)):
				files = glob2.glob(pathname+re[k])

				for t in range(len(files)):
					if files[t] not in d:
						d[files[t]] = num

    with open('categoryLabels.txt','w') as f:
        pickle.dump(d,f)
