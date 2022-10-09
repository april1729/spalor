
Robust Principle Component Analysis
===================================

While PCA is a powerful technique, it has been shown to be less reliable
when just a sparse set of data points are grossly corrupted, and so the
goal of RPCA is to identify and remove such corruptions by separating
the data matrix into the sum of a low rank and sparse matrix. For
example, consider the low rank matrix from the matrix completion example
with a few entries changed

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
           

Formally, we pose this as the nonconvex optimization problem:
:raw-latex:`\begin{equation}
            \begin{array}{ll}
                 \underset{L,S\in \mathbb{R}^{d_1,d_2}}{\text{minimize }}&  \text{rank}(L)+\lambda_0 ||S||_0\\
                 \text{subject to} & L+S=M
            \end{array}
        \end{equation}`

