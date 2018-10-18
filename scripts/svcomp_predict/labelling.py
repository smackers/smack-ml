#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:27:55 2018

@author: ankit
"""

import glob2, sys, csv


class Classify():
    def __init__(self,path):
        self.pathname = path
        
    def readRE(self):
        setAll = glob2.glob(self.pathname+'*.set')
        f = open('label.csv','w')
        for i in range(len(setAll)):
            num = self.labelassign(setAll[i])
            
            if num != None:
                re = [re.rstrip('\n') for re in open(setAll[i])]
                #print(len(re[-1]))
                for k in range(len(re)):
                    if len(re[k]) != 0:
                        files = glob2.glob(self.pathname+re[k])
                    for t in range(len(files)):
                        f.write(files[t]+' '+str(num)+'\n')
                        
        
    def labelassign(self, filename):
        if 'ReachSafety' in filename:
            if 'Arrays' in filename: return 0;
            if 'BitVectors' in filename: return 1;
            if 'ControlFlow' in filename: return 2;
            if 'ECA' in filename: return 3;
            if 'Floats' in filename: return 4;
            if 'Heap' in filename: return 5;
            if 'Loops' in filename: return 6;
            if 'ProductLines' in filename: return 7;
            if 'Recursive' in filename: return 8;
            if 'Sequentialized' in filename: return 9;
            if 'DeviceDriversLinux64' in filename: return 10;
        else:
            num = None
