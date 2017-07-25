import glob2

xml_2 = glob2.glob('../../xmls/di_flag_off/*.xml')
xml_final = []

for filename in xml_2:
    filename1 = filename[:len(filename)-4]
    addition = '-diflag-OFF-allVars-ON'
    filename2 = filename[-4:]
    filename = filename1 + addition + filename2
    xml_final.append(filename)

print xml_final
