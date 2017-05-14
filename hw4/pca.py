from scipy import misc
import numpy as np

subjects = ["A","B","C","D","E","F","G","H","I","J"]
indexes = ["00","01","02","03","04","05","06","07","08","09"]
height = 64
width = 64
faces = []
for sub in subjects:
	for idx in indexes:
		file = "p1/"+sub+idx+".bmp"
		face = misc.imread(file)
		faces.append(face)

faces = np.asarray(faces,dtype="float64")
faces = faces.reshape((len(faces),width*height))
mean = np.mean(faces,axis = 0)
faces_mean = faces - mean
covar = np.cov(np.transpose(faces_mean))
eig_val,eig_vec = np.linalg.eig(covar)
count = 0
eig_vec = np.transpose(eig_vec)
eig_vec = np.real(eig_vec)

w = eig_vec[0:60]
transformed = faces_mean.dot(w.T)
reconstruct = transformed.dot(w)
reconstruct += mean

total_error = 0
for origin,face in zip(faces,reconstruct):
    k = np.sum((origin-face)**2)/len(origin)
    total_error += k
    
total_error /= 100
total_error = np.sqrt(total_error)/255
print (total_error)




