with open('features_tool1.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content]
print len(content), len(content[0])
print content[0]
