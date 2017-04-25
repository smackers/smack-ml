#!/usr/bin/env python
import glob2

'''
	the purpose here is to seprate the '.c', '.i' and '.xml' files into 3 separate files 
	from /data in /scripts/txt. The files generated here can be used to parse the data directly
	
	use 'c_files_script.sh' to run the tool on '.c' files to generate feature vectors

	use 'xml_files_cf.txt' & 'xml_files_h.txt' to parse the xml files and generate the
		appropriate labels for the feature vectors (look at labels_extraction.py next)
'''		

#create different types of files
def open_new_file(filename,mode):
	if mode == 1:
		f = open(filename,'r')	#read mode
	if mode == 2:
		f = open(filename,'w')	#write mode
	if mode == 3:
		f = open(filename,'a+')	#append mode
	return f

#parameter declarations
c_files = glob2.glob('../../../data/c/**/*.c')
i_files = glob2.glob('../../../data/c/**/*.i')
xml_files_cf = glob2.glob('../../xmls/control_flow/*.xml')
xml_files_h = glob2.glob('../../xmls/heap/*.xml')
t = 4
File = [0]*t

#opening the necessary files with appropriate file extension
File[0] = open_new_file('../txt/c_files.txt',2)
File[1] = open_new_file('../txt/xml_files_cf.txt',2)
File[2] = open_new_file('../txt/xml_files_h.txt',2)
File[3] = open_new_file('../txt/i_files.txt',2)

#WRITING all the *.c filenames INTO THE selected FILE
if c_files != None:
	p = 1
	for files in c_files:
		files = files[8:]
		File[p-1].write(files)
		File[p-1].write('\n')

if i_files != None:
	p = 4
	for files in i_files:
		files = files[8:]
		File[p-1].write(files)
		File[p-1].write('\n') 

#WRITING all the *.xml filenames INTO THE selected FILE
if xml_files_cf != None:
	p = 2
	for files in xml_files_cf:
		#files = files[15:]
		File[p-1].write(files)
		File[p-1].write('\n')

if xml_files_h != None:
	p = 3
	for files in xml_files_h:
		#files = files[15:]
		File[p-1].write(files)
		File[p-1].write('\n')


#closing all the open files
for i in range(t):
	File[i].close()
