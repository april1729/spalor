import numpy as np
from spalor.models.cx import CX
A=np.random.randn(500,10).dot(np.random.randn(10,100))


cx_model=CX()
cx_model.fit(A)

Acx=cx_model.C.dot(cx_model.X)

print(np.linalg.norm(Acx-A)/np.linalg.norm(A))