from sklearn.model_selection import train_test_split
import numpy as np
import skvideo.io


# skvideo.utils.rgb2gray
def load_video(file="videos/fish_swarm.avi"):
    data = skvideo.io.vread(file)
    
    data = data[:, :, :, 1]
    return data


def load_movielens(test_size=0.2):
    data = np.loadtxt('movieLens/ratings.dat', delimiter='::')
    X = data[:, [0, 1]].astype(int) - 1
    y = data[:, 2]

    n_users = max(X[:, 0]) + 1
    n_movies = max(X[:, 1]) + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test, n_users, n_movies
