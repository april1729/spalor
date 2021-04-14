import numpy as np

def group_sparse_regression_CX(A,c,eps=1e-8,max_iter=1000):
	(d1,d2)=A.shape
	AtA=A.transpose().dot(A)
	X=np.random.randn(d2,d2)
	I=np.eye(d2)
	res=1;
	for iter in range(0,max_iter):
		res0=res
		g=AtA.dot(X-I)
		step_size=np.sum(np.sum(np.multiply(A.dot(g),A-A.dot(X))))/np.linalg.norm(A.dot(g),ord='fro')**2
		X=X+ step_size*g
		row_norms=np.sum(np.multiply(X,X), axis=1)
		threshhold=np.sort(row_norms)[-c]-0.001
		for row in range(0,d2):
			if row_norms[row]<threshhold:
				X[row,:]=np.zeros(X[row,:].shape)
			else:
				X[row,:]=X[row,:]*(row_norms[row]-threshhold)/row_norms[row]
		res=np.linalg.norm(A-A.dot(X),ord='fro')/np.linalg.norm(A,ord='fro')
		if abs(res-res0)<eps:
			break
	cols=np.where(row_norms>threshhold)	
	C=np.squeeze(A[:,cols])
	X=np.squeeze(X[cols,:])
	return (C,X,cols)