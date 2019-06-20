import os, glob2, subprocess
import resource

def set_limit():
	try:
		byte_lim = 40*(2**30) #40 GB
		resource.setrlimit(resource.RLIMIT_DATA, (byte_lim, byte_lim))
	except ValueError:
		pass

if __name__=='__main__':
	path = os.getenv('HOME')+'/sv-benchmarks/c/'
	fileC = glob2.glob(path+'**/*.c')
	fileI = glob2.glob(path+'**/*.i')
	allFoles = fileC+fileI #python list append
	
	f = open('output_roles.txt','w')
	g = open('output_metrics.txt','w')

	for i in range(len(allFiles)):
		try:
			subprocess.check_call(['./tool1.sh',allFiles[i],'output_metrics.txt','output_roles.txt'],preexec_fn=set_limit)
		except subrocess.CalledProcessError:
			print("Exceeded set limit for ",allFiles[i])
		else:
			print("Stayed within limits for ",allFiles[i])
