import Matrix
import glob2
import Labels
import glob2
import TestCases
import pickle

# ---- assign appropriate unroll value
def UnrollVal(filename):
	if 'unroll1' in filename:
		unroll_code = 1
	elif 'unroll2' in filename:
		unroll_code = 2
	elif 'unroll4' in filename:
		unroll_code = 4
	elif 'unroll6' in filename:
		unroll_code = 6
	elif 'unroll8' in filename:
		unroll_code = 8
	elif 'unroll16' in filename:
		unroll_code = 16
	elif 'unroll32' in filename:
		unroll_code = 32
	elif 'unroll64' in filename:
		unroll_code = 64
	return unroll_code

# ----
tempL0 = Labels.GenerateLabel()
fm0 = Matrix.FinalMatrix()
tc0 = TestCases.Algorithms()
unroll = [1,2,4,6,8,16,32,64]

xml_files = glob2.glob('../../xmls/unroll/*.xml')

CputimeAll = {}
# ---- creating temp collection of all cputime where category = 'correct'
for filen in xml_files:
	code = UnrollVal(filen)
	CputimeAll = tempL0.xmlTree(code,filen,CputimeAll)

LabelsMinCputime = {}
LabelsMinCputime = tempL0.ComputeFinalLabels(CputimeAll,LabelsMinCputime,unroll)

#loading the feature dictionary
Features = pickle.load(open('../txt/FinalFeatures.txt','r'))

'''organize the label class and create the label dictionary.
pass the feature dict and label dict to the matrix function'''


ResultMatrix = fm0.FeatureAndLabelMatrix(Features,LabelsMinCputime)
#print len(ResultMatrix)
tc0.TestingAlgorithmResults(ResultMatrix)
