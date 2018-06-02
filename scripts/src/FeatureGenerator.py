#     ****  (Need not be generated again unless running on different files ****
import FileSeparation
import Features
import pickle
import glob2

fs = FileSeparation.FileExtension()
fg = Features.FeatureGeneration()

n = input("Press '1' to test the algoritms \n Press '2' to test new files")
if n == 1:
	merged_dict = fg.MainFunctionality('../../../data/c/**','../txt/features_tool1.txt','../txt/output_metrics1.txt','../txt/features_tool2.txt')
	# ---- writing the final features to a file (use pickle.load() to load this file directly)
	with open('../txt/FinalFeatures.txt','w') as f:
		pickle.dump(merged_dict,f)
elif n == 2:
	NewTestfilename = raw_input(print "Enter the path of the filenames you want to predict labels for (current_path = home directory): ")
	merged_dict = fg.MainFunctionality(NewTestfilename,'../txt/NewFeaturesTool1.txt','../txt/NewOutputMetrics.txt','../txt/NewFeaturesTool2.txt')
	# ---- writing the final features to a file (use pickle.load() to load this file directly)
	with open('../txt/NewFinalFeatures.txt','w') as f:
		pickle.dump(merged_dict,f)
