import FileSeparation
import Features
import pickle
import glob2

########  (Need not be generated again unless running on different files) ########

fs = FileSeparation.FileExtension()
fg = Features.FeatureGeneration()

# ---- separated files
c_files = glob2.glob('../../../data/c/**/*.c')
i_files = glob2.glob('../../../data/c/**/*.i')

filenames = c_files + i_files
#print type(c_files_merged)

#fs.write_new_file(open('../txt/c_files.txt','w'),c_files_merged,8)


# ---- (uncomment this section for) generating all features from scratch
for filename in filenames:
	filename = '../../..'+filename
	fg.RunTool1(filename)
	fg.RunTool2(filename)

# ---- formatting and merging of features generated above
with open('../txt/features_tool1.txt','r') as f:
	content1 = f.readlines()

content_tool1 = [x.strip() for x in content1]
feature_dict_tool1 = fg.Formatting(content_tool1, len(content_tool1))
#print feature_dict_tool1

with open('../txt/features_tool2.txt','r') as g:
	content2 = g.readlines()

content_tool2 = [x.strip() for x in content2]
feature_dict_tool2 = fg.Formatting(content_tool2,len(content_tool2))
#print feature_dict_tool2


merged_dict = fg.MergeFeatures(feature_dict_tool1, feature_dict_tool2)
#print merged_dict

# ---- writing the final features to a file
with open('../txt/FinalFeatures.txt','w') as f:
	pickle.dump(merged_dict,f)