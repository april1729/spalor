Matrix Completion for movie recomendations
##################################


Theres a ton of applications of matrix completion, from reconstructing
the molecular structure of protiens from limited measurements to image
classification, but by far the most commonly cited example is the
Netflix problem. The state of the art dataset for movie recommendations
comes from `MovieLens <https://grouplens.org/datasets/movielens/>`__,
and though they have datasets with 25 million ratings, we’re going to
stick with 1 million for simplicity.

Before we get into the data, we should justify to ourslelves that this
is going to be a low-rank matrix. Let’s take the movies Breakfast Club
and Pretty in Pink as an example. I would bet that the way individuals
rate these two movies is pretty much the same way, and so they columns
associated with each of them should be very close to eachother. Now lets
throw Titanic into the mix. While I wouldn’t expect it to be the same,
it might be similiar. It might also be similiar to other period pieces
featuring forbidden love, like Pride and Prejudice, or movies with
Leonardo DeCaprio, like Wolf of Wallstreet. So, I would expect that the
ratings for Titanic might look like an average of all of these movies.
The point is that the ratings for a specific movie should be pretty
close to a linear combination of ratings of just a few other movies.

First, lets load the data set and see what it looks like.

.. code:: ipython3

    data = np.loadtxt( 'movieLens/ratings.dat',delimiter='::' )
    print(data[:][0:3])


.. parsed-literal::

    [[1.00000000e+00 1.19300000e+03 5.00000000e+00 9.78300760e+08]
     [1.00000000e+00 6.61000000e+02 3.00000000e+00 9.78302109e+08]
     [1.00000000e+00 9.14000000e+02 3.00000000e+00 9.78301968e+08]]


The first column is the user ID, the second is the movie ID, the third
is the rating (1,2,3,4, or 5), and the last is a time stamp (which we
don’t need to worry about). We want the rows of the matrix to be users,
and the columns should be movies.

.. code:: ipython3

    X=data[:, [0,1]].astype(int)-1
    y=data[:,2]
    
    n_users=max(X[:,0])+1
    n_movies=max(X[:,1])+1
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


.. parsed-literal::

    (6040, 3952)


Now to train the model and make some predictions

.. code:: ipython3

    from spalor.MC import MC
    from statistics import mean
    
    mc_model=MC(n_users,n_movies,r=5)
    mc_model.fit(X_train, y_train)
    y_predict=mc_model.predict(X_test)
    
    print("MAE:",mean(abs(y_test-y_predict)))
    print("Percent of predictions off my less than 1: ",np.sum(abs(y_test-y_predict)<1)/len(y_test))



.. parsed-literal::

    MAE: 0.6910439339771605
    Percent of predictions off my less than 1:  0.7603603243318903

