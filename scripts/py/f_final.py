import pickle

'''
Purpose: Formatting of the features generated on the entire data
   Note: This need not be generated again unless adding more input data points
'''

with open('../txt/output_metrics.txt','r') as f:
	content = f.readlines()

content = [x.strip() for x in content]

temp_list = []
feature_dict = {}

# ---- list of list containing the filename & features vectors
for i in range(len(content)):
	listed = content[i].strip().split(' ')
	temp_list.append(listed)

# ---- convert the feature vector values 'string' ---> 'float'.
# ---- feature_dict = {filename: [feature-vector]}
for i in range(len(temp_list)):
	temp = map(float,temp_list[i][1:])
	if len(temp) == 20:	
		feature_dict[temp_list[i][0]] = temp

# ---- writing to a file
with open('../txt/list_of_features.txt','w') as f:
	pickle.dump(feature_dict,f)

