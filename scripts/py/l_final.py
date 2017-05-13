import pickle

'''
	Here, using the '.txt' files and computing the unroll with min(cputime) 
	and 'status' == 'correct' meaning the correct label.
	If 'status' == wrong || error, then return unroll 1.

	The output is the unroll value with min(cputime) for all the files in just one dictionary (min_cputime_unroll).
'''

def compute_final_labels(f,min_cputime_unroll):
	unroll_values = [1,2,4,6,8,16,32,64]
	unroll_values = {}
	for filename in f:
		for i in range(len(f[filename])):
			if f[filename][i] == 'correct' or f[filename][i] == 'wrong' or f[filename][i] == 'error':
				if filename in min_cputime_unroll:
					min_cputime_unroll[filename].append(f[filename][i-1])	#write the cputime for all status
				else:
					min_cputime_unroll[filename] = []
					min_cputime_unroll[filename].append(f[filename][i-1])
		#---------------
		temp = min(min_cputime_unroll[filename])	#computing the min from the list above
	
		for i in range(len(min_cputime_unroll[filename])):	#finding the mean in the list
			if temp == min_cputime_unroll[filename][i]:
				print i
				min_cputime_unroll[filename] = unroll_values[i]	#assigning the 'index value + 1' to get the unroll
				break
		#---------------
		'''
		Use the following section to do the marked section optimally. Instead of searching the min again in the list
	
		for i in range(len(min_cputime_unroll[filename])):
			min_cputime_unroll[filename] = (min_cputime_unroll[filename].index(min(min_cputime_unroll[filename]))) + 1
		'''
	
	return min_cputime_unroll

#reading the data
g = pickle.load(open('../txt/labels_all.txt','rb'))
min_cputime_unroll = {}

compute_final_labels(g,min_cputime_unroll)

print min_cputime_unroll
print len(min_cputime_unroll)

#writing the final labels unroll for [min(cputime) & category == 'correct']
with open('../txt/final_labels.txt','w') as f:
	pickle.dump(min_cputime_unroll,f)
