#!/usr/bin/env python
import glob2, pickle
pathname = 'input_SMACK'; extension = 'c'
files = glob2.glob(pathname + '/*.' + extension)

filen = open('InputFile.txt','w')

for item in files:
	if 'true-unreach' in item: 
		print>>filen,item 