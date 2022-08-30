# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import random
import tflearn
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scipy import spatial

from transformers import BertTokenizer, BertModel

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# Load data
print('loading data...')
data = pd.read_csv('BERT_training_set2.csv')

# Embeddings Dict
print('creating embeddings dictionary...')
embeddings = pd.read_csv('/Users/kalliehuynh/compound-word-embeddings/BERT_embeddings.csv')
embeddings_dict = {}
for i in embeddings.index:
    embeddings_dict[embeddings.iloc[i, 1]] = np.array(embeddings.iloc[i, 2:], dtype='float32')

# embeddings_dict = {}
# with open('/Users/kalliehuynh/compound-word-embeddings/BERT_embeddings.csv', 'r') as file:
#     content = file.readlines()
#     for line in content:
#         if line:
#             split_line = line.split()
#             word = split_line[0]
#             vec = np.asarray(split_line[1:], dtype='float32')
#             if word in valid_words or word in cp_words:  
#                 embeddings_dict[word] = vec
# print('embeddings dictionary complete!')

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
 
def closest_cosine_similarity(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding))    


rows = len(data)

df = data.copy()
df = df.sample(frac=1).reset_index(drop=True)


np_array = np.array(df)
train_rows = round(len(df)*0.7)
test_rows = rows  - train_rows

trainX = np.array(df.iloc[:train_rows, 4:1540], dtype="float32")  # The vectors for the constituent words
set_trainX = set(df.iloc[:train_rows, 4:1540])

trainY = np.array(df.iloc[:train_rows, 1540:], dtype="float32")  # The vector for the compound word

testX = np.array(df.iloc[train_rows:, 4:1540], dtype="float32")
testY = np.array(df.iloc[train_rows:, 1540:], dtype="float32")
print('testX shape:', testX.shape)
print('testY shape:', testY.shape)

print('loading data complete!')

# Reshape the input for LSTM
print('data reshaping...')
trainX = trainX.reshape((train_rows, 2, 768))
testX = testX.reshape(testX.shape[0], 2, 768)
print('data reshaping complete!')

def cosine_similarity(y_pred, y_true):
    with tf.name_scope("Similarity"):
        dot_product = tf.reduce_sum(tf.math.multiply(y_pred,y_true),axis=1)

        pred_magnitudes = tf.sqrt(tf.reduce_sum(tf.math.multiply(y_pred,y_pred),axis=1))
        true_magnitudes = tf.sqrt(tf.reduce_sum(tf.math.multiply(y_true,y_true),axis=1))

        cos = dot_product/(pred_magnitudes * true_magnitudes)

        return tf.math.abs(tf.reduce_mean(cos) - 1)

def mean_square(y_pred, y_true):
    """ Mean Square Loss.

    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets (labels).

    """
    with tf.name_scope("MeanSquare"):
        return tf.reduce_mean(tf.square(y_pred - y_true))

def custom_loss(y_pred, y_true):
    alpha = 0.05
    cosine_loss = cosine_similarity(y_pred, y_true)
    mean_square_loss = mean_square(y_pred, y_true)

    loss = tf.cond(cosine_loss<alpha,
                            lambda: 0.75*cosine_loss + 0.25*mean_square_loss,
                            lambda: cosine_loss)
    return loss

# Network building
batch_size = 4
learning_rate = 0.0005
print('network building...')
net = tflearn.input_data(shape=[None, 2, 768])  
net = tflearn.lstm(net, 768) 
net = tflearn.layers.core.dropout(net, 0.70)
net = tflearn.fully_connected(net, 768) 
net = tflearn.regression(net, optimizer='adam', batch_size=batch_size, learning_rate=learning_rate, metric='R2', loss='custom_loss')


# Training
print('training...')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('/Users/kalliehuynh/compound-word-embeddings/BERT_models/lstm_BERT_modelx.tflearn')
# model.fit(trainX, trainY, validation_set=(testX, testY), n_epoch=150, show_metric=True)

# Save the model
# model.save('/Users/kalliehuynh/compound-word-embeddings/BERT_models/BERT_model301.tfl')

# Results
result = model.evaluate(testX, testY)
print("test acc:", result)


def generate_TSNE_all(predicted, true):
    """Generates a TSNE visualization of the predicted embeddings and the true embeddings
        Predicted embeddings are in coral, true are in light blue.
    Args:
        predicted : predicted embeddings
        true : true embeddings
    """
    model = TSNE(2)
    transformed_p = model.fit_transform(predicted)
    transformed_t = model.fit_transform(true)

    p_xs = transformed_p[:,0]
    p_ys = transformed_p[:,1]

    t_xs = transformed_t[:,0]
    t_ys = transformed_t[:,1]

    fig = plt.figure()

    plt.subplot(1, 3, 1)
    plt.scatter(t_xs, t_ys, c='lightblue')  # true are light blue

    plt.subplot(1, 3, 2)
    plt.scatter(p_xs, p_ys, c='coral')  # predicted are coral

    plt.subplot(1, 3, 3)
    plt.scatter(t_xs, t_ys, c='lightblue')  # true are light blue
    plt.scatter(p_xs, p_ys, c='coral')  # predicted are coral

    plt.show()




# Visible testing
samples = 3
print("Generate predictions for ", samples, " samples")
predictions = model.predict(testX[:samples])
print("predictions shape:", predictions.shape)
print('')
print('')

print('PREDICTED EMBEDDING')
for i in range(samples):
    closest_words = find_closest_embeddings(predictions[i])[:5] # Closest 5 words for each prediction
    print(df.loc[train_rows + i, ['c1', 'c2', 'cmp']], closest_words)
    # print(predictions[i])
    print('closest cosine distance to the prediction:', end=' ')
    print(spatial.distance.cosine(predictions[i], embeddings_dict[closest_words[0]]), end=', ')
    print(spatial.distance.cosine(predictions[i], embeddings_dict[closest_words[1]]))
print('')

# Generate TSNE viualization
# For all embeddings
labels = model.predict(testX)
words = data.iloc[train_rows:, 2]
features = np.array(data.iloc[train_rows:, 1540:], dtype="float32")

generate_TSNE_all(labels, features)

print('BATCH SIZE:', batch_size)
print('LEARNING_RATE:', learning_rate)

print(tf.__file__)

