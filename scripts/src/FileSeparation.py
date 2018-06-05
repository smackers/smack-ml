#!/usr/bin/env python
import glob2

'''
	the purpose here is to seprate the '.c', '.i' and '.xml' files into 3 separate files 
	from /data in /scripts/txt. The files generated here can be used to parse the data directly

	use 'xml_files.txt' to parse the xml files and generate the
		appropriate labels for the feature vectors
'''		
class FileExtension(object):
	def __init__(self):
		pass

	def write_new_file(self,filename, temp_files,i):
		# ---- WRITING all the *.c filenames INTO c_files.txt
		# ---- 'i' is the trim length since it's different for '.c' and '.xml' files
		if temp_files != None:
			for files in temp_files:
				files = files[i:]
				filename.write(files)
				filename.write('\n')

		# ---- closing all the open files
		filename.close()

