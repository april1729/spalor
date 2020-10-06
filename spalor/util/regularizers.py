import numpy as np



# Scad functions
def scad(x, gamma, a):
    return 0


def scad_sg(x, gamma, a):
    return 0


def scad_prox(x, gamma, a, beta):
    return 0


#MCP Functions
def mcp(x, gamma, a):
    return min(gamma*abs(x)-x^2/(2*a), gamma^2 * a/2)


def mcp_sg(x, gamma, a):
    return 0


def mcp_prox(x, gamma, a, beta):
    if abs(x)< a * gamma:
        return np.sign(x)*((a*beta)/(a*beta-1)) * (abs(x)-gamma/beta)
    else:
        return x


#L1 functions
def l1(x, gamma, a):
    return gamma*abs(x)


def l1_sg(x, gamma, a):
    return gamma*np.sign(x)


def l1_prox(x, gamma, a, beta):
    return np.sign(x)*(abs(x)-gamma/beta)
