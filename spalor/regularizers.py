import numpy as np



# Scad functions
def scad(x, gamma, a):
    return 0


def scad_sg(x, gamma, a):
    return 0


def scad_prox(x, gamma, a):
    return 0


#MCP Functions
def mcp(x, gamma, a):
    return min(gamma*abs(x)-x^2/(2*a), gamma^2 * a/2)


def mcp_sg(x, gamma, a):
    return 0


def mcp_prox(x, gamma, a):
    y=np.multiply(np.sign(x),((a)/(a-1)) * np.maximum(abs(x)-gamma,0))
    y[abs(x)>a*gamma]=x[abs(x)>a*gamma]
    return y

#L1 functions
def l1(x, gamma, a):
    return gamma*abs(x)


def l1_sg(x, gamma, a):
    return gamma*np.sign(x)


def l1_prox(x, gamma, a):
    return np.multiply(np.sign(x),(abs(x)-gamma))