#!/usr/bin/env python
import glob2

'''
	the purpose here is to seprate the '.c', '.i' and '.xml' files into 3 separate files 
	from /data in /scripts/txt. The files generated here can be used to parse the data directly
	
	Run 'extraction_f.py' after this file 

	use 'xml_files.txt' to parse the xml files and generate the
		appropriate labels for the feature vectors
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

#separated files
c_files = glob2.glob('../../../data/c/**/*.c')
i_files = glob2.glob('../../../data/c/**/*.i')
xml_files = glob2.glob('../../xmls/*.xml')

t = 3
File = [0]*t

# ---- opening the necessary files with appropriate file extension
File[0] = open_new_file('../txt/c_files.txt',2)
File[1] = open_new_file('../txt/i_files.txt',2)
File[2] = open_new_file('../txt/xml_files.txt',2)

# ---- WRITING all the *.c filenames INTO c_files.txt
if c_files != None:
	p = 1
	for files in c_files:
		files = files[8:]   #truncate to match filenames later (while merging the labels with features)
		File[p-1].write(files)
		File[p-1].write('\n')

# ---- WRITING all the *.i filenames INTO i_files.txt
if i_files != None:
	p = 2
	for files in i_files:
		files = files[8:]
		File[p-1].write(files)
		File[p-1].write('\n') 

# ---- WRITING all the *.xml filenames INTO xml_files.txt
if xml_files != None:
	p = 3
	for files in xml_files:
		File[p-1].write(files)
		File[p-1].write('\n')

# ---- closing all the open files
for i in range(t):
	File[i].close()
