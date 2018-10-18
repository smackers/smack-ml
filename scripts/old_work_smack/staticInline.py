import glob2
import Labels
import Matrix
import pickle
import TestCases

# --- Setting the diFlag value
def FlagVal(filename):
    if 'AllVars-staticInlining-ON' in filename:
		code = 1
    elif 'AllVarsOFF-staticInliningON' in filename:
		code = 2
    return code

AllXml = glob2.glob('../../xmls/staticInline/*.xml')

tempL1 = Labels.GenerateLabel()
fm1 = Matrix.FinalMatrix()
tc1 = TestCases.Algorithms()
di_flag = [1,2]
CputimeAll = {}

for filen in AllXml:
	code = FlagVal(filen)
	CputimeAll = tempL1.xmlTree(code,filen,CputimeAll)

LabelMinCputime = {}
LabelMinCputime = tempL1.ComputeFinalLabels(CputimeAll,LabelMinCputime,di_flag)

'''temp_count = 0
for i in LabelMinCputime:
	if LabelMinCputime[i] == 1:
		temp_count += 1

print temp_count, len(LabelMinCputime)

print LabelMinCputime
print LabelMinCputime['../../c/array-examples/data_structures_set_multi_proc_false-unreach-call_ground.c']
print LabelMinCputime['../../c/array-examples/data_structures_set_multi_proc_false-unreach-call_ground..i']

with open('../txt/diFinalLabels.txt','w') as f:
	pickle.dump(LabelMinCputime,f)'''

#loading the feature dictionary
Features = pickle.load(open('../txt/FinalFeatures.txt','r'))

'''organize the label class and create the label dictionary.
pass the feature dict and label dict to the matrix function'''

ResultMatrix = fm1.FeatureAndLabelMatrix(Features,LabelMinCputime)
#print len(ResultMatrix)
tc1.TestCasesForAlgorithm(ResultMatrix, None)
