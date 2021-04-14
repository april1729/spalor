import numpy as np



def rand_low_rank_mat(d1,d2,r):
	M=np.random.randn(d1,r).dot(np.random.randn(r,d2))
	return M
def rand_sparse_mat(d1,d2,alpha):
	s=round(alpha*d1*d2)
	S=np.zeros((d1,d2))
	S[np.random.choice(d1,size=s), np.random.choice(d2,size=s)]=np.random.choice([-1,1],size=s)
	return S
def mc_test(d1,d2,r,p):
	n=round(d1*d2*p)
	M=rand_low_rank_mat(d1,d2,r)
	X=np.concatenate((np.random.randint(d1, size=(n,1)), np.random.randint(d2, size=(n,1))), axis=1)
	y=M[X[:,0],X[:,1]]
	return (M,X,y)

def rpca_test(d1,d2,r,alpha):
	L=rand_low_rank_mat(d1,d2,r)
	S=rand_sparse_mat(d1,d2,alpha)
	return (L,S)

def lasso_test(n,d,s):
	A=np.random.randn(n,d)
	x=rand_sparse_mat(d,1,s)
	y=A.dot(y)
	return (A,x,y)