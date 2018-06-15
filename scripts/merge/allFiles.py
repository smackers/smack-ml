import glob2, pickle

'''Goal: generate list of all SVCOMP benchmarks (2017)'''
if __name__ == '__main__':
    path = '/proj/SMACK/sv-benchmarks/c'
    cfiles = glob2.glob(path + '/**/*.c')
    ifiles = glob2.glob(path + '/**/*.i')
    all_files = cfiles + ifiles

    with open('allBenchmarks.txt','w') as f:
        pickle.dump(all_files,f)
