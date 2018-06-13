import glob2

# generate file with </foldername/file_name>
def collect_file(path):
    cfiles = glob2.glob(path + '/**/*.c')
    ifiles = glob2.glob(path + '/**/*.i')
    f = open('all_files.txt','w')
    count = 0
    #print len(cfiles), len(ifiles)
    #print cfiles[0], ifiles[0]

    for i in range(len(cfiles)):
        f.write(cfiles[i][27:]+'\n')
        if i < len(ifiles): f.write(ifiles[i][27:]+'\n');
    f.close()

if __name__ == '__main__':
    path = '/proj/SMACK/sv-benchmarks/c'
    print len(path)
    collect_file(path)
