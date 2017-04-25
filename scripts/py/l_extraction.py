from xml.dom.minidom import parse
import xml.dom.minidom
import pickle

'''
This section is for single xml file
Format = Filename:[xml_code,unroll #, <status> value, <cputime> value,<category> value]'''

def data_structure(property_for,result,unroll_cputime,i):

	#find <tag> = 'run' in the .xml files
	runs = result.getElementsByTagName('run')
	print len(runs)
	for run in runs:
		#check if 'run' has attributes, 'name' and 'properties' and then begin the data-structure
		if run.hasAttribute('name') and run.hasAttribute('properties'):
			filename = str(run.getAttribute('name'))
			filename = filename[16:]

			if filename in unroll_cputime:
				unroll_cputime[filename].append(property_for)
			else:			
				unroll_cputime[filename] = []
				unroll_cputime[filename].append(property_for)
			
			unroll_cputime[filename].append(i)	# 'i' is the unroll value
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

#===============================    creating a list of .xml files   ======================================================
content_cf = [line.rstrip('\n') for line in open('../txt/xml_files_cf.txt')]
content_cf.sort()

content_h = [line.rstrip('\n') for line in open('../txt/xml_files_h.txt')]
content_h.sort()

#output dictionary declaration
unroll_cputime_cf = {}
unroll_cputime_h = {}

''' creating 2 tree roots for 2 different .xml files'''

for i in range(len(content_cf)):
	#root-1 for cf files
	DOMTree_1 = xml.dom.minidom.parse(content_cf[i])
	result_1 = DOMTree_1.documentElement
	property_for = 00
	unroll_cputime_cf = data_structure(property_for,result_1,unroll_cputime_cf,i+1)
	
	#root-2 for heap files
	DOMTree_2 = xml.dom.minidom.parse(content_h[i])
	result_2 = DOMTree_2.documentElement
	property_for = 01
	unroll_cputime_h = data_structure(property_for,result_2,unroll_cputime_h,i+1)


#print len(unroll_cputime_cf)
#print len(unroll_cputime_h)
#writing to a file
with open('../txt/labels_cf.txt','w') as f:
	pickle.dump(unroll_cputime_cf,f)
with open('../txt/labels_h.txt','w') as f:
	pickle.dump(unroll_cputime_h,f)
