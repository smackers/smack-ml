import subprocess
import shlex

#sending the filenames to run the tool to generate the feature vectors
def calling_shell_script(input_file,output_roles,output_metrics):
		subprocess.call(['bash','. ../sh/c_files_script.sh',input_file,output_roles,output_metrics])


#list of all files in the .txt files
filenames = [files.rstrip('\n') for files in open('../txt/i_files.txt')]
filenames2 = [files.rstrip('\n') for files in open('../txt/c_files.txt')]

for filename in filenames:
	filename_roles = '../txt/output_roles.txt'
	filename_metrics = '../txt/output_metrics.txt'
	filename = '../../../data/c/'+filename
	calling_shell_script(filename,filename_roles,filename_metrics)


for filename2 in filenames2:
	filename_roles = '../txt/output_roles.txt'
	filename_metrics = '../txt/output_metrics.txt'
	filename2 = '../../../data/c/'+filename2
	calling_shell_script(filename2,filename_roles,filename_metrics)
