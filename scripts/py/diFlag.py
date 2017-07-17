import glob2
import Labels
import Matrix
import pickle
import TestCases

# --- Setting the diFlag value
def FlagVal(filename):
	if 'di_file_on' in filename:
		code = 0
	else:
		code = 1
	return code


xml_1 = glob2.glob('../../xmls/di_flag_on/*.xml')
xml_2 = glob2.glob('../../xmls/di_flag_off/*.xml')

AllXml = xml_1 + xml_2

tempL1 = Labels.GenerateLabel()
fm1 = Matrix.FinalMatrix()
tc1 = TestCases.Algorithms()
di_flag = [0,1]
CputimeAll = {}

for filen in AllXml:
	code = FlagVal(filen)
	CputimeAll = tempL1.xmlTree(code,filen,CputimeAll)

LabelMinCputime = {}
LabelMinCputime = tempL1.ComputeFinalLabels(CputimeAll,LabelMinCputime,di_flag)

#loading the feature dictionary
Features = pickle.load(open('../txt/FinalFeatures.txt','r'))

'''organize the label class and create the label dictionary.
pass the feature dict and label dict to the matrix function'''


ResultMatrix = fm1.FeatureAndLabelMatrix(Features,LabelMinCputime)
print len(ResultMatrix)
tc1.TestingAlgorithmResults(ResultMatrix)
