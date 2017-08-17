import glob2
import Labels
import Matrix
import pickle
import TestCases

# --- Setting the diFlag value
def FlagVal(filename):
	if 'diflagOFF-allVarsON' in filename:
		code = 1
	elif 'diflagON-allVarsON' in filename:
		code = 2
	elif 'diflag-AllVars-OFF' in filename:
		code = 3
	elif 'diflagON-AllVarsOFF' in filename:
		code = 4
	return code


xml_1 = glob2.glob('../../xmls/diFlag/*.xml')
xml_2 = glob2.glob('../../xmls/allVars/*.xml')
AllXml = xml_1 + xml_2

tempL1 = Labels.GenerateLabel()
fm1 = Matrix.FinalMatrix()
tc1 = TestCases.Algorithms()
trackAll = [1,2,3,4]
CputimeAll = {}

for filen in AllXml:
	code = FlagVal(filen)
	CputimeAll = tempL1.xmlTree(code,filen,CputimeAll)

LabelMinCputime = {}
LabelMinCputime = tempL1.ComputeFinalLabels(CputimeAll,LabelMinCputime,trackAll)

'''temp_count = 0
for i in LabelMinCputime:
	if LabelMinCputime[i] == 1:
		temp_count += 1

print temp_count, len(LabelMinCputime)'''
#loading the feature dictionary
Features = pickle.load(open('../txt/FinalFeatures.txt','r'))

'''organize the label class and create the label dictionary.
pass the feature dict and label dict to the matrix function'''

ResultMatrix = fm1.FeatureAndLabelMatrix(Features,LabelMinCputime)
print len(ResultMatrix)
tc1.TestCasesForAlgorithm(ResultMatrix, None)
