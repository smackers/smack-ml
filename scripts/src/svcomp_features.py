import pickle


f = open('../txt/FinalFeatures.txt','r')
features = pickle.load(f)

#print len(features), len(features[0])
print features
