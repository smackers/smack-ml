from xml.dom.minidom import parse
import xml.dom.minidom
import pickle

'''
			Purpose: Extract status, cputime, category for all files in all the *.xml files.
				Data structure: Dictionary {'filename':[above 3 values]}


This section is for single xml file
Format = Filename:[xml_code,unroll #, <status> value, <cputime> value,<category> value]'''
def data_structure(property_for,result,unroll_cputime):

	#find <tag> = 'run' in the .xml files
	runs = result.getElementsByTagName('run')

	for run in runs:
		#check if 'run' has attributes, 'name' and 'properties' and then begin the data-structure
		if run.hasAttribute('name') and run.hasAttribute('properties'):
			filename = str(run.getAttribute('name'))
			filename = '../../c/' + filename[16:]   #to do a type-match of filename for merging with feature vectors

			if filename in unroll_cputime:
				unroll_cputime[filename].append(property_for)
			else:			
				unroll_cputime[filename] = []
				unroll_cputime[filename].append(property_for)
			
			#unroll_cputime[filename].append(i)	# 'i' is the unroll value
			#unroll_cputime[filename].append(check)

		#find <tag> = 'column' as the child of <tag> = 'run'
		columns = run.getElementsByTagName('column')
		
		for column in columns:
			
			if str(column.getAttribute('title')) == 'status':
				unroll_cputime[filename].append(str(column.getAttribute('value')))

			#find value of 'status' and 'cuptime' and append to list
			if str(column.getAttribute('title')) == 'cputime':
				#convert the value from string to float and rounding the value
				value = str(column.getAttribute('value'))
				value = round(float(value[:len(value)-1]),10)
			
			if str(column.getAttribute('title')) == 'category':
				face_value = str(column.getAttribute('value'))
				
				if face_value == 'correct':
					unroll_cputime[filename].append(value)
				else:
					unroll_cputime[filename].append(500000000)	#face_value = ('wrong' || 'TIMEOUT')
				unroll_cputime[filename].append(face_value)

	return unroll_cputime	

def unroll_code(filename):
	
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



#===============================    creating a list of .xml files   ======================================================
content = [line.rstrip('\n') for line in open('../txt/xml_files.txt')]
#content.sort()  #sorting so that the unrolls are written in order (not needed though)
#print content
#output dictionary declaration
unroll_cputime = {}

''' creating tree root for .xml files'''

for i in range(len(content)):
	
	DOMTree_1 = xml.dom.minidom.parse(content[i])
	result_1 = DOMTree_1.documentElement
	property_for = unroll_code(content[i])
	unroll_cputime = data_structure(property_for,result_1,unroll_cputime)

print unroll_cputime	
print len(unroll_cputime)
#writing to a file
with open('../txt/labels_all.txt','w') as f:
	pickle.dump(unroll_cputime,f)
