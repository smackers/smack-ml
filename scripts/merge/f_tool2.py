import sys, os

#Purpose: Create a dictionary with a type-match key as labels
def Formatting(self, file, SplitParameter):
    dic = {}; benchpath = '/proj/SMACK/sv-benchmarks/';
    with open(file,'r') as f:
    			content1 = f.readlines()

    		content_tool1 = [x.strip() for x in content1]

    if SplitParameter == ' ':
        for i in range(n):
            listed = content[i].strip().split(SplitParameter)
            temp = map(float,listed[1:])
            #if the features vectors are not of equal length (precaution fo	r feature matrix)
            if len(temp) == 20:
                listed[0] = benchpath + listed[0][7:]
                dic[listed[0]] = temp
    elif SplitParameter == '\t':
        for i in range(n):
            listed = content[i].strip().split(SplitParameter)
            temp = map(float,listed[1:])
            if len(temp) == 13:
                listed[0] = benchpath + listed[0][15:]
                dic[listed[0]] = temp
    return dic


if __name__ == '__main__':
    file1 = 'f_tool2a.txt'
    file2 = 'f_tool2b.txt'
