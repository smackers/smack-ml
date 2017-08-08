class FinalMatrix(object):
	def __init__(self):
		pass

	'''Purpose: creating the matrix with features & labels'''
	def FeatureAndLabelMatrix(self,files_f,files_l):
		vector_l = []
		matrix_f = []

		# ---- matrix_f = [feature_vector, label]
		for f in files_l:
			if f in files_f:
				files_f[f].append(files_l[f])
				if len(files_f[f]) == 34:	#needed to make sure the matrix is generated correctly (sometimes the tools doesn't generate feature vectors of same length)
					matrix_f.append(files_f[f])

		'''with open("../txt/f_matrix.txt","w") as f:
			pickle.dump(matrix_f,f)'''
		return matrix_f
