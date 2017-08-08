import Matrix, Features, Labels, TestCases
import glob2, pickle

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
tempL = Labels.GenerateLabel()
fmatrix = Matrix.FinalMatrix()
testc = TestCases.Algorithms()
Newfeature = Features.FeatureGeneration()
unroll = [1,2,4,6,8,16,32,64]

xml_files = glob2.glob('../../xmls/unroll/*.xml')

CputimeAll = {}
# ---- creating temp collection of all cputime where category = 'correct'
for filen in xml_files:
	code = UnrollVal(filen)
	CputimeAll = tempL.xmlTree(code,filen,CputimeAll)

LabelsMinCputime = {}
LabelsMinCputime = tempL.ComputeFinalLabels(CputimeAll,LabelsMinCputime,unroll)

#loading the feature dictionary
Features = pickle.load(open('../txt/FinalFeatures.txt','r'))

'''organize the label class and create the label dictionary.
pass the feature dict and label dict to the matrix function'''


ResultMatrix = fmatrix.FeatureAndLabelMatrix(Features,LabelsMinCputime)
#print len(ResultMatrix)
k = input(print "Choose a number from the following options \n 1. To Test Algorithms \n 2. To predict label for a new file")

if k == 1:
	testc.TestCasesForAlgorithm(ResultMatrix)
elif k == 2:
	NewTestfilename = raw_input(print "Enter the path of the filename you want to predict: ")
	Newfeature.RunTool1(NewTestfilename)
	Newfeature.RunTool2(NewTestfilename)
	feature_vector_tool1 = Newfeature.formatting`

	tco.
