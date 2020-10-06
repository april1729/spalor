

```python
from IPython.display import Image
from IPython.display import display_html
from IPython.display import display
from IPython.display import Math
from IPython.display import Latex
from IPython.display import HTML
```

<h1> What is Matrix Completion? </h1>
<p> Simply put, the goal of matrix completion is fill in missing entries of a matrix (or dataset) given the fact that the matrix is low rank, or low dimensional.  Essentially, it's like a game of Sudoku with a different set of rules. Lets say I have a matrix that I know is supposed to be rank 2.  That means that every column can be written as a linear combination (weighted sum) of two vectors.  Lets look at an example of what this puzzle might look like.  </p>

$$ \begin{bmatrix}   
1 & 1 &2 & 2\\
2&1&3&\\
1&2&&1\\
\end{bmatrix}$$

<p> The first two columns are completly filled in, so we can use those to figure out the rest of the columns.  Based on the few entries in the third column that are given, we can see that the third column should probably be the first column plus the second column.  Likewise, the fourth column is two times the first column minus the second column. </p>
    
$$ \begin{bmatrix}   
1 & 1 &2 & 2\\
2&1&3&5\\
1&2&3&1\\
\end{bmatrix}$$

<p> That was a particularly easy example since we knew the first two columns completely. </p>  

    
<p> To see why we should care about this, here's a claim that shouldn't be too hard to believe: <b> Datasets are inherently low rank </b>.  In the example we just did, the columns could be movies, the rows could be people, and the numbers could be how each person rated each movie.  Obviously, this is going to be sparse since not everyone has seen every movie.  That's where matrix completions comes in.  When we filled in the missing entries, we gave our guess as to what movies people are going to enjoy. After explaining an algorithm to do matrix completion, we're going to try this for a data set with a million ratings people gave movies and see how well we recommend movies to people.</p>
  
<h1> How do we do it? </h1>

There's two paradigms for matrix completion.  One is to minimize the rank of a matrix that fits our measurements, and the other is to find a matrix of a given rank that matches up with our known entries.  In this blog post, I'll just be talking about the second.  

Before we explain the algorithm, we need to introduce a little more notation. We are going to let $\Omega$ be the set of indices where we know the entry.  For example, if we have the partially observed matrix
$$ \begin{matrix}
\color{blue}1\\\color{blue}2\\\color{blue}3
\end{matrix}
\begin{bmatrix}   
  & 1 &  \\
  &   & 1\\
1 &   &  
  \end{bmatrix}$$
    
  $$ 
\begin{matrix}   
 &\color{red}1 & \color{red}2 & \color{red}3  \end{matrix}$$
then, $\Omega$ would be $\{ (\color{blue} 1, \color{red}2), (\color{blue}2 , \color{red}3),(\color{blue} 3, \color{red}1)\}$  We can now pose the problem of finding a matrix with rank $r$ that best fits the entries we've observe as an <i> optimization problem</i>.  
$$
\begin{align}
&\underset{X}{\text{minimize}}& \sum_{(i,j)\text{ in }\Omega} (X_{ij}-M_{ij})^2 \\
& \text{such that} & \text{rank}(X)=r
\end{align}
$$
The first line specifies <i> objective function </i>(the function we want to minimize), which is the sum of the square of the difference between $X_{ij}$ and $M_{ij}$ for every $(i,j)$ that we have a measurement for.  The second line is our <i> constraint</i>, which says that the matrix has to be rank $r$.

While minimizing a function like that isn't too hard, forcing the matrix to be rank $r$ can be tricky. One property of a low rank matrix that has $m$ rows and $n$ columns is that we can factor it into two smaller matrices like such: 
$$X=UV$$
where $U$ is $n$ by $r$ and $V$ is $r$ by $m$.  So now, if we can find matrices $U$ and $V$ such that the matrix $UV$ fits our data, we know its going to be rank $r$ and that will be the solution to our problem. 

If $u_i$ is the $i^{th}$ column of $U$ and $v_j$ is the $j^{th}$ column of $V$, then $X_{ij}$ is the <i> inner product </i> of $u_i$ and $v_j$, $X_{ij}= \langle u_i, v_i \rangle$.  We can rewrite the optimization problem we want to solve as 
$$
\begin{align}
&\underset{U, V}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u_i, v_i \rangle-M_{ij})^2 
\end{align}
$$
In order to solve this, we can alternate between optimizing for $U$ while letting $V$ be a constant, and optimizing over $V$ while letting $U$ be a constant.  If $t$ is the iteration number, then the algorithm is simply 
$$
\begin{align}
\text{for } t=1,2,\ldots:& \\
U^{t}=&\underset{U}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u_i, v^{t-1}_i \rangle-M_{ij})^2 \\
V^{t}=&\underset{ V}{\text{minimize}}& \sum_{(i,j)\in \Omega} (\langle u^t_i, v_i \rangle-M_{ij})^2 \\
\end{align}
$$
At each iteration, we just need to solve a least squares problem which is easy enough.  





```python
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
```

Lets test our algorithm with the simple example given earlier.


```python
X=([0,0,0,0,1,1,1,2,2,2], [0,1,2,3,0,1,2,0,1,3])
y=[1,1,2,2,2,1,3,1,2,1]
(U,V)=alt_min(3,4,2,X, y)
print(U.dot(V))
```

    [[1.00007939 0.99998808 1.9999993  2.00000032]
     [1.99997349 1.00000394 3.00000041 4.99908422]
     [0.99997352 2.00000397 2.99974998 0.99999984]]


Thats the same matrix we came up with!
## SpaLoR: A python package for Sparse and Low Rank models
While matrix completion is a form of machine learning, I always see it talked about and coded up as just an optimization algorithm. To help bridge this gap, I've been working on a python package for called [SpaLoR](https://github.com/april1729/SpaLoR), which should be easy to pick up for scikit-learn users. 

In it, there is a model called <i>MatrixCompletion</i>, which you can use to fit your data to a low rank matrix and then make predictions.  No parameters are required when you initialize, but a few you might want to specify are
 - <i>method</i>: there are a few different methods you could use. 
  - <i>AltMin</i>:
  - <i>NonconvexReg</i>:
  - <i>NuclearNorm</i>:
 - <i> eps </i>: the average amount of error you would expect in any individual entry.  The algorithm will pick a rank as small as possible so that the average error is no larger than <i>eps</i>.  If you make it too small, you're more likely to overfit your data, and if you make it too large, the 
 - <i>r</i>: an upper bound on the rank of the matrix.  Making this smaller will make the algorithm run faster.
 



<h1> How do we use it for movie recomendations? </h1>

Now that we have a good understanding of what matrix completion is and how to do it, we can get to the fun part.  Theres a ton of applications of matrix completion, from reconstructing the molecular structure of protiens from limited measurements(LINK) to image classification(LINK), but by far the most commonly cited example is the Netflix problem.  The state of the art dataset for movie recommendations comes from MovieLens, and though they have datasets with 25 million ratings, we're going to stick with 1 million for simplicity.  

First, lets load the data set and see what it looks like.


```python
data = np.loadtxt( 'movieLens/ratings.dat',delimiter='::' )
print(data[:][0:3])
```

    [[1.00000000e+00 1.19300000e+03 5.00000000e+00 9.78300760e+08]
     [1.00000000e+00 6.61000000e+02 3.00000000e+00 9.78302109e+08]
     [1.00000000e+00 9.14000000e+02 3.00000000e+00 9.78301968e+08]]


The first column is the user ID, the second is the movie ID, the third is the rating (1,2,3,4, or 5), and the last is a time stamp (which we don't need to worry about).  We want the rows of the matrix to be users, and the columns should be movies. 


```python
X=data[:, [0,1]].astype(int)-1
y=data[:,2]

n_users=max(X[:,0])+1
n_movies=max(X[:,1])+1

print((n_users,n_movies))
```

    (6040, 3952)


So, we have 6040 users and 3952 movies.  That's a total of about 23 million potential ratings, of which we know 1 million.  We're going to reserve 200,000 of the ratings to test our results.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Now to train the model and make some predictions!


```python
from MC import *
from statistics import mean

mc_model=MC(n_users,n_movies,5)
mc_model.fit(np.array(X_train).transpose(), y_train)
y_predict=mc_model.predict((np.array(X_test).transpose()))

print("MAE:",mean(abs(y_test-y_predict)))
print("Percent of predictions off my less than 1: ",np.sum(abs(y_test-y_predict)<1)/len(y_test))
```

    MAE: 0.6910439339771605
    Percent of predictions off my less than 1:  0.7603603243318903


I would say 0.691 is pretty good for the mean absolute error.  Plus, 76% of the predictions are off by less than 1.

These numbers can most definetly get better.  Here's a few ideas I might write about in the future to get better results:
 - Use nonnegative matrix completion. This uses the more restrictive rule that every row is a nonnegative combination of other rows.  
 - Include data about genres.  We could have a column thats just 1 if a movie is a horror movie and 0 if it isn't.  Then, we have a fully observed column which will help train our model and make more informed predictions.
