import pickle

'''
	Purpose: using the 'labels_all.txt' file and computing the unroll with min(cputime) 
	and 'status' == 'correct' meaning the correct label.

	min_cputime_unroll = {filename: unroll for min(cputime)}
'''

# ---- returns a dictionary with filename & the best possible unroll (based on min(cputime))
def compute_final_labels(f,min_cputime_unroll):
	unroll_values = [1,2,4,6,8,16,32,64]
	for filename in f:
		# ---- initialization
		if filename not in min_cputime_unroll:
			min_cputime_unroll[filename] = [500000]*8
		# ---- compute the min and check for unroll overlapping
		for i in range(len(f[filename])):
			if f[filename][i] == 'correct' or f[filename][i] == 'wrong' or f[filename][i] == 'error':
				t = f[filename][i-3]	#unroll value for the above condition
				
				# ---- making sure that only the min cputime for each unroll is appended to the list
				for j in range(len(unroll_values)):
					if t == unroll_values[j]:
						new_min = min_cputime_unroll[filename][j]
						if f[filename][i-1] < new_min: 
							min_cputime_unroll[filename][j] = f[filename][i-1]
							
		temp = min(min_cputime_unroll[filename])	#compute the min_cputime

		# ---- check if there is a min_cputime: if not, then assign the highest unroll value by default
		if temp != 500000:
			k = min_cputime_unroll[filename].index(temp)
			min_cputime_unroll[filename] = unroll_values[k]
		else:
			min_cputime_unroll[filename] = 0
	
	return min_cputime_unroll

# ---- reading the data
g = pickle.load(open('../txt/labels_all.txt','rb'))
min_cputime_unroll = {}

min_cputime_unroll = compute_final_labels(g,min_cputime_unroll)

# ---- writing the final labels unroll {filename: unroll for min(cputime)}
with open('../txt/final_labels.txt','w') as f:
	pickle.dump(min_cputime_unroll,f)
