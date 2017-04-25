import pickle

with open('../txt/output_metrics.txt','r') as f:
	content = f.readlines()

content = [x.strip() for x in content]
temp_list = []
f_matrix = []
feature_dict = {}

#list of list containing the filename & features vectors
for i in range(len(content)):
	listed = content[i].strip().split(' ')
	temp_list.append(listed)

#convert the feature vector values 'string' ---> 'float'.
for i in range(len(temp_list)):
	feature_dict[temp_list[i][0]] = map(float,temp_list[i][1:])
	f_matrix.append(feature_dict[temp_list[i][0]])
	
#writing to a file
with open('../txt/list_of_features.txt','w') as f:
	pickle.dump(feature_dict,f)

with open('../txt/features_matrix.txt','w') as g:
	pickle.dump(f_matrix,g)
