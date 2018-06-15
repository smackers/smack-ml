#!/usr/bin/env python2.7
import pickle
import numpy as np

'''Goals:
1. make lists
2. remove junk from file name and rename with path = '/proj/SMACK/sv-benchmarks/c'
3. Convert into dictionary to merge lists/ merge lists directly
4. Store into a file <path_new = path + '/output/file_name'>
'''

class fix_filename():
    def __init__(self, fname, path, k):
        self.fname = fname
        self.path = path
        self.k = k

    def make_lists(self,splitParameter,values):
        new_content = []
        with open(self.fname,'r') as f:
			content = f.readlines()
        '''Goal1: make lists'''
        content = [x.strip().split(splitParameter) for x in content]

        '''Goal2: remove junk and rename'''
        for i in range(len(content)):
            tmp2 = []
            if len(content[i][1:]) == values:
                tmp = self.path + content[i][0][self.k:]
                new_content.append([tmp] + map(float,content[i][1:]))

        #print len(new_content), len(new_content[0])
        return new_content

class Merge():
    def __init__(self):
        pass

    '''Goal3: Merging the features'''
    def merge_lists(self, list1, list2):
        df = {}
        for i in range(len(list1)):
            for j in range(len(list2)):
                if list1[i][0] == list2[j][0]:
                    df[list1[i][0]] = list1[i][1:] + list2[j][1:]
        return df

    '''Merge Labels with corresponding feature vectors'''
    def merge_labels(self, df1, dl1):
        merged_dl = []; fnameVec = [];
        for item in df1:
            if item in dl1:
                fnameVec.append(item)
                merged_dl.append(df1[item]+[dl1[item]])
        return merged_dl, fnameVec

if __name__ == '__main__':
    #path = sys.argv[1] #pass the path to the benchmarks
    path = '/proj/SMACK/sv-benchmarks/c/';

    file1 = fix_filename('f_tool2a.txt',path,8)
    f1 = file1.make_lists(' ',20)

    file2 = fix_filename('f_tool2b.txt',path,16)
    f2 = file2.make_lists('\t',13)

    m = Merge()
    merged_df = m.merge_lists(f1,f2)

    '''Goal4: Storing the feature vectors'''
    with open('notNormalizedMergedFeatures.txt','w') as f:
        pickle.dump(merged_df,f)

    dl = pickle.load(open('categoryLabels.txt','r'))
    merged_all,fnames = m.merge_labels(merged_df,dl)
    merged_all = np.matrix(merged_all)
    #print merged_all.shape

    '''Goal4: Storing the final numpy.matrix()'''
    with open('notNormalizedMergedLabels.txt','w') as f:
        pickle.dump(merged_all,f)
