'''Purpose: Extract filename, status, cputime, category for all files in all the *.xml files'''
#run file: python <filename> <benchmark_type>

from xml.dom.minidom import parse
import xml.dom.minidom
import pickle, glob2, sys

# ---- Format = Filename:[xml_code,unroll #, <status> value, <cputime> value,<category> value]
def data_structure(result):
	# ---- find <tag> = 'run' in the .xml files
	runs = result.getElementsByTagName('run')

	for run in runs:
		# ---- check if 'run' has attributes, 'name' and 'properties' and then begin the data-structure
		if run.hasAttribute('name'):
			filename = str(run.getAttribute('name'))
			vector = path + filename[21:]

		# ---- find <tag> = 'column' as the child of <tag> = 'run'
		columns = run.getElementsByTagName('column')

		for column in columns:

			if str(column.getAttribute('title')) == 'Aliasing':
				vector = vector + ' ' + str(column.getAttribute('value'))
			if str(column.getAttribute('title')) == 'Arrays':
				vector = vector + ' ' + str(column.getAttribute('value'))
			if str(column.getAttribute('title')) == 'Boolean':
				vector = vector + ' ' + str(column.getAttribute('value'))
			if str(column.getAttribute('title')) == 'Composite types':
				vector = vector + ' ' + str(column.getAttribute('value'))
			
		f.write(vector + '\n')
#===============================    creating a list of .xml files   ======================================================
#content = [line.rstrip('\n') for line in open('xml_files.txt')]
#finding the .xml file to extract all filenames for respective benchmark
#benchmark = str(sys.argv[1])
path = '/proj/SMACK/sv-benchmarks/c'
content = glob2.glob('*.xml')
print(content)

f = open('featureVec.txt','w')
# ---- creating tree root for .xml files
DOMTree = parse(content[0])
result = DOMTree.documentElement
data_structure(result)

'''
g = open(benchmark+"/all"+benchmark+".txt",'r')
test_content = g.readlines()
for i in range(0,len(test_content),3):
	f.write(test_content[i])
'''
