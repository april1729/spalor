Robust Principle Component Analysis
===================================

While PCA is a powerful technique, it’s less reliable when just a sparse
set of data points are grossly corrupted, and so the goal of RPCA is to
identify and remove outliers by separating the data matrix into the sum
of a low rank and sparse matrix. For example, consider the low rank
matrix from the matrix completion example with a few entries changed

.. math::

   \begin{bmatrix}  
           1 &\color{purple}{\textbf{17}}& 3 & 4\\ 
           3 & 6 &\color{purple}{\textbf{7}}& 12 \\
           5 & 10 & 15  & \color{purple}{\textbf{2}} \\
           7 & \color{purple}{\textbf{3}} & 21 & 28 \\
           \end{bmatrix}
           =
           {\begin{bmatrix}  
           1 & 2 & 3 & 4\\ 
           3 & 6 & 9 & 12 \\
           5 & 10 & 15  & 20 \\
           7 & 14 & 21 & 28 \\
           \end{bmatrix}}
           +{
           \begin{bmatrix}  
           & -15 &  & \\ 
            &  &  -2&  \\
            &  &   &  18\\
            & 11 &  &  \\
           \end{bmatrix}}
           

RPCA solves the nonconvex optimization problem:

.. math::

   \begin{equation}
               \begin{array}{ll}
                    \underset{L,S\in \mathbb{R}^{d_1,d_2}}{\text{minimize }}&  \text{rank}(L)+\lambda_0 ||S||_0\\
                    \text{subject to} & L+S=M
               \end{array}
           \end{equation}

The ``RPCA`` class
------------------

.. code:: ipython3

    import numpy as np
    from spalor.models import RPCA
    A = np.random.randn(50, 1).dot(np.random.randn(1,30))
    S = np.random.rand(*A.shape)<0.1
    
    rpca=RPCA(n_components=1, sparsity=0.1)
    rpca.fit(A+S)
    
    print("Denoised matrix error: \n", np.linalg.norm(rpca.to_matrix()-A)/np.linalg.norm(A))
    print("Outliersm error: \n", np.linalg.norm(rpca.outliers_-S)/np.linalg.norm(S))


.. parsed-literal::

    Denoised matrix error: 
     4.94329075927598e-16
    Outliersm error: 
     4.510225048268804e-16

