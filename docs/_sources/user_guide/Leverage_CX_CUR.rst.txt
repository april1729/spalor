CUR and CX decomposions
=======================

Given a large, low rank matrix, CX decomposition selects a subset of
columns and finds a representation of the original matrix in terms of
the selected columns. For a given matrix, :math:`A` with :math:`n` rows
and :math:`d` columns, the decomposition is, :math:`A=CX` where
:math:`C` is a :math:`d \times n_c` matrix, whose columns are all
columns in :math:`A`, and :math:`X` is a :math:`n_c \times n` that
minimizes :math:`||A-CX||_F^2`.

For example, consider the following matrix:

.. math::

    \begin{bmatrix}
   1 & 1 &2 & 2\\
   2&1&3&5\\
   1&2&3&1\\
   3 & 1 & 4 & 8
   \end{bmatrix}

If we chose the first two columns as :math:`C`, the CX decomposition
would be:

.. math::

   C=\begin{bmatrix}
   1 & 1\\
   2&1\\
   1&2\\
   3 & 1\\
   \end{bmatrix}, X=\begin{bmatrix}
   1 & 0 & 1& 3 \\
   0 & 1 & 1& -1 \\
   \end{bmatrix}

`CUR
decompositions <https://www.pnas.org/doi/epdf/10.1073/pnas.0803205106>`__
work in a very similiar way, but also selects a subset of rows in
addition to columns. For a given matrix, :math:`A` with :math:`n` rows
and :math:`d` columns, the decomposition is, :math:`A=CUR` where
:math:`C` is a :math:`d \times n_c` matrix, whose columns are all
columns in :math:`A`, and :math:`R` is a :math:`n_r \times n` whose rows
are all rows in :math:`A`, and :math:`U` is the :math:`n_c \times n_r`
matrix that minimized that minimizes :math:`||A-CUR||_F^2`.

Using the same example, a CUR decomposition of :math:`A` could be:

.. math::

   C=\begin{bmatrix}
   1 & 1\\
   2&1\\
   1&2\\
   3 & 1
   \end{bmatrix},
   R=\begin{bmatrix}
   1 & 1 &2 & 2\\
   2&1&3&5\\
   \end{bmatrix} , U=\begin{bmatrix} -1 & 1\\ 2 & -1\end{bmatrix}

CX and CUR decompositions can be done with any set of rows or columns.
One way to select rows and columns is by calculating the `leverage
score <https://en.wikipedia.org/wiki/Leverage_(statistics)>`__ of each
row and columns. The leverage score is a measure of how important each
row (or column) is when constructing the low-rank approximation of a
matrix. Once the leverage scores have been calculated, we can randomly
sample rows (or columns) using the leverage scores as probabilities.

Calculated the leverage scores exactly requires computing the `truncated
SVD <https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD>`__,
but the can be
`approximated <https://www.stat.berkeley.edu/~mmahoney/pubs/coherence-jmlr12.pdf>`__
quickly and accurately.

The ``CX`` class
----------------

:class:``CX``

.. code:: ipython3

    import numpy as np
    from spalor.models import CX
    
    A=np.array([[1, 1, 2, 2],
                [2, 1, 3, 5],
                [1, 2, 3, 1],
                [3, 1, 4, 8]])
    
    cx=CX()
    X=cx.fit_transform(A, cols=[0,1])
    print("C:\n", cx.C)
    print("X:\n", X)
    print("columns used: \n", cx.cols)


.. parsed-literal::

    C:
     [[1 1]
     [2 1]
     [1 2]
     [3 1]]
    X:
     [[1 1]
     [2 1]
     [1 2]
     [3 1]]
    columns used: 
     [0, 1]


.. code:: ipython3

    cx=CX(n_components=2)
    X=cx.fit_transform(A)
    print("C:\n", cx.C)
    print("X:\n", X)
    print("columns used: \n", cx.cols)


.. parsed-literal::

    C:
     [[2 2]
     [5 5]
     [1 1]
     [8 8]]
    X:
     [[2 2]
     [5 5]
     [1 1]
     [8 8]]
    columns used: 
     [3 3]


The ``CUR`` class
-----------------

:class:``CUR``

.. code:: ipython3

    import numpy as np
    from spalor.models import CUR
    
    A=np.array([[1, 1, 2, 2],
                [2, 1, 3, 5],
                [1, 2, 3, 1],
                [3, 1, 4, 8]], dtype=float)
    
    cur = CUR(n_rows=2, n_cols=2)
    cur.fit(A)
    
    print("C:\n", cur.C)
    print("U:\n", cur.U)
    print("R:\n", cur.R)
    print("columns used: \n", cur.cols)
    print("rows used: \n", cur.rows)


.. parsed-literal::

    C:
     [[2. 2.]
     [3. 5.]
     [3. 1.]
     [4. 8.]]
    U:
     [[-0.05  0.4 ]
     [ 0.15 -0.2 ]]
    R:
     [[3. 1. 4. 8.]
     [1. 2. 3. 1.]]
    columns used: 
     [array([2, 3])]
    rows used: 
     [array([3, 2])]


Computing leverage sores
------------------------

.. code:: ipython3

    from spalor.matrix_tools import leverage_score
    import numpy as np
    A=np.array([[1, 1, 2, 2],
                [2, 1, 3, 5],
                [1, 2, 3, 1],
                [3, 1, 4, 8]], dtype=float)
    
    print(leverage_score(A, k=2, axis=1))


.. parsed-literal::

    [0.05172414 0.18965517 0.31034483 0.44827586]


.. code:: ipython3

    svdA=np.linalg.svd(A)
    print(leverage_score(svdA, k=2, axis=1))


.. parsed-literal::

    [0.05172414 0.18965517 0.31034483 0.44827586]


.. code:: ipython3

    print(leverage_score(A, k=2, axis=1, method="approximate"))


.. parsed-literal::

    [0.08855471 0.00874692 0.09747291 0.80522546]

