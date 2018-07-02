#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:28:28 2018

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

#4 Compute SVD
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

#5 Calculate cosine similarity, sort by most similar and return the top N.
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
              movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])
        
#6 Select k principal components to represent the movies, a movie_id to find recommendations and print the top_n results.
k = 50
movie_id = 1 # Grab an id from movies.dat
top_n = 10

sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, indexes)