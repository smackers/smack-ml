from xml.dom.minidom import parse
import xml.dom.minidom
import pickle
'''
			Purpose: Extract status, cputime, category for all files in all the *.xml files.
				Data structure: Dictionary {'filename':[above 3 values]}'''


class GenerateLabel(object):
	def __init__(self):
		pass

	# ---- create a tree structure for the xml doc
	def xmlTree(self,code,xmlDoc,temp_dict):
		DOMTree_1 = xml.dom.minidom.parse(xmlDoc)
		result = DOMTree_1.documentElement

		# ---- find <tag> = 'run' in the .xml files
		runs = result.getElementsByTagName('run')

		for run in runs:
			# ---- check if 'run' has attributes, 'name' and 'properties' and then begin the data-structure
			if run.hasAttribute('name') and run.hasAttribute('properties'):
				filename = str(run.getAttribute('name'))
				filename = '../../c/' + filename[16:]   #to do a type-match of filename for merging with feature vectors

				if filename in temp_dict:
					temp_dict[filename].append(code)
				else:
					temp_dict[filename] = []
					temp_dict[filename].append(code)

			# ---- find <tag> = 'column' as the child of <tag> = 'run'
			columns = run.getElementsByTagName('column')

			for column in columns:
				if str(column.getAttribute('title')) == 'status':
					temp_dict[filename].append(str(column.getAttribute('value')))

				# ---- find value of 'status' and 'cuptime' and append to list
				if str(column.getAttribute('title')) == 'cputime':
					# ---- convert the value from string to float and rounding the value
					value = str(column.getAttribute('value'))
					value = round(float(value[:len(value)-1]),10)

				if str(column.getAttribute('title')) == 'category':
					face_value = str(column.getAttribute('value'))

					if face_value == 'correct':
						temp_dict[filename].append(value)
					else:
						temp_dict[filename].append(500000)	#face_value = ('wrong' || 'error')
					temp_dict[filename].append(face_value)

		return temp_dict


	'''
		Purpose: using the 'labels_all.txt' file and computing the unroll with min(cputime)
		and 'status' == 'correct' meaning the correct label.

		min_cputime = {filename: unroll for min(cputime)}
	'''

	# ---- returns a dictionary with filename & the best possible unroll (based on min(cputime))
	def ComputeFinalLabels(self,f,min_cputime,testLength):
		for filename in f:
			# ---- initialization
			if filename not in min_cputime:
				min_cputime[filename] = [500000]*len(testLength)
			# ---- compute the min and check for unroll overlapping
			for i in range(len(f[filename])):
				if f[filename][i] == 'correct' or f[filename][i] == 'wrong' or f[filename][i] == 'error':
					t = f[filename][i-3]	#unroll value for the above condition

					# ---- making sure that only the min cputime for each unroll is appended to the list
					for j in range(len(testLength)):
						if t == testLength[j]:
							new_min = min_cputime[filename][j]
							if f[filename][i-1] < new_min:
								min_cputime[filename][j] = f[filename][i-1]

			temp = min(min_cputime[filename])	#compute the min_cputime

			# ---- check if there is a min_cputime: if not, then assign the highest unroll value by default
			if temp != 500000:
				k = min_cputime[filename].index(temp)
				min_cputime[filename] = testLength[k]
			else:
				min_cputime[filename] = 0

		return min_cputime
