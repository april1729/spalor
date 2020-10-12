import numpy as np

def partXY(x,y, known):
    return np.sum(np.multiply(x[known[0][:], :],y[known[1][:],:]), axis=1)

