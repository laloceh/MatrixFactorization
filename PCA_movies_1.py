#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:52:22 2018

@author: eduardo
https://alyssaq.github.io/2015/20150426-simple-movie-recommender-using-svd/

"""

import numpy as np
import pandas as pd

#1  Read the files with pandas
data = pd.io.parsers.read_csv('data/ratings.dat', 
                              names=['user_id', 'movie_id', 'rating', 'time'],
                                engine='python', delimiter='::')

movie_data = pd.io.parsers.read_csv('data/movies.dat',
                                    names=['movie_id', 'title', 'genre'],
                                    engine='python', delimiter='::')

print data.head(4)
print movie_data.head(4)

#2  Create the ratings matrix of shape (m×u) with rows as movies and columns as users
ratings_mat = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

#3 Normalise matrix (subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

#4 computing PCA using the eigenvectors of the co-variance matrix
normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
cov_mat = np.cov(normalised_mat)
evals, evecs = np.linalg.eig(cov_mat)

#5 Select k principal components to represent the movies, a movie_id to find recommendations and print the top_n results.
k = 50
movie_id = 1 # Grab an id from movies.dat
top_n = 10

sliced = evecs[:, :k] # representative data
top_indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, top_indexes)