import pickle

files_f = pickle.load(open('../txt/list_of_features.txt','rb'))
files_l = pickle.load(open('../txt/final_labels.txt','rb'))
vector_l = []
matrix_f = []
k = 150

for f in files_l:
	if f in files_f:
		matrix_f.append(files_f[f])
		vector_l.append(files_l[f])

'''
print matrix_f
print vector_l,
print len(matrix_f), len(vector_l)
'''

training_matrix = matrix_f[:k]
training_labels = vector_l[:k]
test_matrix = matrix_f[k:]
test_labels = vector_l[k:]

with open('../txt/training/matrix.txt','w') as f:
	pickle.dump(training_matrix,f)
with open('../txt/training/label.txt','w') as g:
	pickle.dump(training_labels,g)
with open('../txt/test/matrix.txt','w') as h:
	pickle.dump(test_matrix,h)
with open('../txt/test/label.txt','w') as j:
	pickle.dump(test_labels,j)
