
.. code:: ipython3

    from IPython.display import Image
    from IPython.display import display_html
    from IPython.display import display
    from IPython.display import Math
    from IPython.display import Latex
    from IPython.display import HTML

.. raw:: html

   <h1>

What is Matrix Completion?

.. raw:: html

   </h1>

.. raw:: html

   <p>

Simply put, the goal of matrix completion is fill in missing entries of
a matrix (or dataset) given the fact that the matrix is low rank, or low
dimensional. Essentially, it’s like a game of Sudoku with a different
set of rules. Lets say I have a matrix that I know is supposed to be
rank 2. That means that every column can be written as a linear
combination (weighted sum) of two vectors. Lets look at an example of
what this puzzle might look like.

.. raw:: html

   </p>

.. math::

    \begin{bmatrix}   
   1 & 1 &2 & 2\\
   2&1&3&\\
   1&2&&1
   \end{bmatrix}

.. raw:: html

   <p>

The first two columns are completly filled in, so we can use those to
figure out the rest of the columns. Based on the few entries in the
third column that are given, we can see that the third column should
probably be the first column plus the second column. Likewise, the
fourth column is two times the first column minus the second column.

.. raw:: html

   </p>

.. math::

    \begin{bmatrix}   
   1 & 1 &2 & 2\\
   2&1&3&5\\
   1&2&3&1\\
   \end{bmatrix}

.. raw:: html

   <p>

That was a particularly easy example since we knew the first two columns
completely.

.. raw:: html

   </p>

.. raw:: html

   <p>

To see why we should care about this, here’s a claim that shouldn’t be
too hard to believe: Datasets are inherently low rank . In the example
we just did, the columns could be movies, the rows could be people, and
the numbers could be how each person rated each movie. Obviously, this
is going to be sparse since not everyone has seen every movie. That’s
where matrix completions comes in. When we filled in the missing
entries, we gave our guess as to what movies people are going to enjoy.
After explaining an algorithm to do matrix completion, we’re going to
try this for a data set with a million ratings people gave movies and
see how well we recommend movies to people.

.. raw:: html

   </p>

.. raw:: html

   <h1>

How do we do it?

.. raw:: html

   </h1>

There’s two paradigms for matrix completion. One is to minimize the rank
of a matrix that fits our measurements, and the other is to find a
matrix of a given rank that matches up with our known entries. Here,
we’re just going to give an example using the latter of the two.

Before we explain the algorithm, we need to introduce a little more
notation. We are going to let :math:`\Omega` be the set of indices where
we know the entry. For example, if we have the partially observed matrix

.. math::

    \begin{matrix}
   \color{blue}1\\\color{blue}2\\\color{blue}3
   \end{matrix}
   \begin{bmatrix}   
     & 1 &  \\
     &   & 1\\
   1 &   &  
     \end{bmatrix}

| 

  .. math::

      
     \begin{matrix}   
      &\color{red}1 & \color{red}2 & \color{red}3  \end{matrix}

   then, :math:`\Omega` would be
  :math:`\{ (\color{blue} 1, \color{red}2), (\color{blue}2 , \color{red}3),(\color{blue} 3, \color{red}1)\}`
  We can now pose the problem of finding a matrix with rank :math:`r`
  that best fits the entries we’ve observe as an optimization problem.
| 

  .. math::


     \begin{align}
     &\underset{X}{\text{minimize}}& \sum_{(i,j)\text{ in }\Omega} (X_{ij}-M_{ij})^2 \\
     & \text{such that} & \text{rank}(X)=r
     \end{align}

   The first line specifies objective function (the function we want to
  minimize), which is the sum of the square of the difference between
  :math:`X_{ij}` and :math:`M_{ij}` for every :math:`(i,j)` that we have
  a measurement for. The second line is our constraint, which says that
  the matrix has to be rank :math:`r`.

While minimizing a function like that isn’t too hard, forcing the matrix
to be rank :math:`r` can be tricky. One property of a low rank matrix
that has :math:`m` rows and :math:`n` columns is that we can factor it
into two smaller matrices like such:

.. math:: X=UV

 where :math:`U` is :math:`n` by :math:`r` and :math:`V` is :math:`r` by
:math:`m`. So now, if we can find matrices :math:`U` and :math:`V` such
that the matrix :math:`UV` fits our data, we know its going to be rank
:math:`r` and that will be the solution to our problem.

If :math:`u_i` is the :math:`i^{th}` column of :math:`U` and :math:`v_j`
is the :math:`j^{th}` column of :math:`V`, then :math:`X_{ij}` is the
inner product of :math:`u_i` and :math:`v_j`,
:math:`X_{ij}= \langle u_i, v_i \rangle`. We can rewrite the
optimization problem we want to solve as

.. math::


   \begin{align}
   &\underset{U, V}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u_i, v_i \rangle-M_{ij})^2 
   \end{align}

 In order to solve this, we can alternate between optimizing for
:math:`U` while letting :math:`V` be a constant, and optimizing over
:math:`V` while letting :math:`U` be a constant. If :math:`t` is the
iteration number, then the algorithm is simply

.. math::


   \begin{align}
   \text{for } t=1,2,\ldots:& \\
   U^{t}=&\underset{U}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u_i, v^{t-1}_i \rangle-M_{ij})^2 \\
   V^{t}=&\underset{ V}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u^t_i, v_i \rangle-M_{ij})^2 \\
   \end{align}

 At each iteration, we just need to solve a least squares problem which
is easy enough.

.. code:: ipython3

    import numpy as np
    from scipy.optimize import minimize
    
    def alt_min(m,n,r, Omega, known):
        U=np.random.rand(m,r)
        V=np.random.rand(r,n)
    
        for i in range(0,100):   
            
            objU=lambda x: np.linalg.norm(np.reshape(x, [m,r]).dot(V)[Omega]-known)**2
            U = np.reshape(minimize(objU, U).x, [m,r])
            
            objV=lambda x: np.linalg.norm(U.dot(np.reshape(x, [r,n]))[Omega]-known)**2
            V = np.reshape(minimize(objV, V).x, [r,n])
    
            res=np.linalg.norm(U.dot(V)[Omega]-known)
            if res < 0.0001:
                break
        return (U,V)

Lets test our algorithm with the simple example given earlier.

.. code:: ipython3

    X=([0,0,0,0,1,1,1,2,2,2], [0,1,2,3,0,1,2,0,1,3])
    y=[1,1,2,2,2,1,3,1,2,1]
    (U,V)=alt_min(3,4,2,X, y)
    print(U.dot(V))


.. parsed-literal::

    [[1.00008887 0.9999849  1.99999906 1.99999915]
     [1.99997047 1.00000478 2.99999981 4.9989627 ]
     [0.99997001 2.0000056  2.99972689 1.00000089]]


Thats the same matrix we came up with!

The MC class in SpaLoR
----------------------

While this function would work just fine for smaller problems, the MC
class in SpaLoR is much more efficient and can handle very large
matrices.

