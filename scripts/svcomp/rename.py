

path = '/proj/SMACK/sv-benchmarks/c/'

with open('f_tool2.txt') as f:
    content = f.readlines()

content = [x.strip().split(' ') for x in content]
print len(content)
for i in range(1,len(content)):
    content[i][0] = path + content[i][0][34:]
    content[i][1:] = map(float,content[i][1:])

print len(content)
