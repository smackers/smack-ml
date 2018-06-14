import numpy as np
from features import processing

if __name__ == '__main__':
    path1 = '/proj/SMACK/sv-benchmarks/c/';
    file1 = processing('f_tool2a.txt',path1,16)
    f1 = file1.make_lists(' ')

    file2 = processing('f_tool2b.txt',path1,16)
    f2 = file2.make_lists('\t')
    #print f2
