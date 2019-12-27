from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
import gzip
import nltk

import tensorflow as tf


class GloveModel():
    def __init__(self, data):
        self.data = data

        # Build a cooccurence matrix (counts where a target is used within the local context)
        self.word_dict, raw_cooccurence_mat = self.__build_raw_cooccurence_matrix()
        self.probability_cooccurence_table = self.__build_probability_matrix(raw_cooccurence_mat)



    def __build_raw_cooccurence_matrix(self):
        word_dict = dict()
        word_array = []
        idx = 0
        for sentence_vector in token_sentence_vecs:
            for i, word in enumerate(sentence_vector):
                if (word not in word_dict.keys()):
                    word_array.append(word)
                    word_dict[word] = idx
                    idx += 1
        cooccurence_dict = {}

        print(word_array)
        # Create a vector map for each word in the corpus
        for sentence_vector in token_sentence_vecs:
            for word in sentence_vector:
                cooccurence_dict[word] = [0.0] * len(word_dict.keys())

        # Build a cooccurence matrix (window_size is hard coded for now to be 2)
        for sentence_vector in token_sentence_vecs:
            for center_word_idx, word in enumerate(sentence_vector):
                if (center_word_idx == 0 or center_word_idx + 1 < len(sentence_vector)):
                    rightOne = sentence_vector[center_word_idx + 1]
                    cooccurence_dict[word][word_dict[rightOne]] += 1.0
                if (center_word_idx == 0 or center_word_idx + 2 < len(sentence_vector)):
                    rightTwo = sentence_vector[center_word_idx + 2]
                    cooccurence_dict[word][word_dict[rightTwo]] += 1.0
                if (center_word_idx - 1 >= 0):
                    leftOne = sentence_vector[center_word_idx - 1]
                    cooccurence_dict[word][word_dict[leftOne]] += 1.0
                if (center_word_idx - 2 >= 0):
                    leftTwo = sentence_vector[center_word_idx - 2]
                    cooccurence_dict[word][word_dict[leftTwo]] += 1.0

        cooccurence_mat = []
        for word in word_array:
            cooccurence_mat.append(cooccurence_dict[word])

        cooccurence_mat = np.array(cooccurence_mat)

        return word_dict, cooccurence_mat


    def __build_probability_matrix(self, cooccurence_mat):
        for i, row in enumerate(cooccurence_mat):
            row_sum = sum(row)
            for j, col in enumerate(row):
                cooccurence_mat[i][j] /= row_sum

        return cooccurence_mat

    def print_prob_table(self):
        print(self.probability_cooccurence_table)

    def generate_tsne(self, path=None, size=(100, 100), word_count=10, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
            from sklearn.manifold import TSNE
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
            low_dim_embeddings = tsne.fit_transform(embeddings[:word_count, :])
            labels = self.words[:word_count]


if __name__ == '__main__':
    example_sentences = ["I want to break my fast", "I like my school", "I am browsing internet",
                         "I will not go to school tomorrow", "School is absolutely boring", "He is going to school"]

    example_sentences2 = ["I like deep learning.", "I like NLP.", "I enjoy flying."]

    token_sentence_vecs = [nltk.word_tokenize(sentence.lower()) for sentence in example_sentences]

    glove = GloveModel(token_sentence_vecs)
    glove.print_prob_table()