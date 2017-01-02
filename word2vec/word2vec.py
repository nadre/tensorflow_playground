import collections
import math
import os
import random
import re
import json
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

def main():
	filename = maybe_download('text8.zip', 31344016)
	words = read_words(filename)
	print('Data size %d' % len(words))

	word2vec = Word2Vec()
	word2vec.train(words)

def maybe_download(filename, expected_bytes):
	"""Download a file if not present, and make sure it's the right size."""
	if not os.path.exists(filename):
		url = 'http://mattmahoney.net/dc/'
		filename, _ = urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified %s' % filename)
	else:
		print(statinfo.st_size)
		raise Exception(
		'Failed to verify ' + filename + '. Can you get to it with a browser?')
	return filename

def read_words(filename):
	"""Extract the first file enclosed in a zip file as a list of words"""
	with zipfile.ZipFile(filename) as f:
		words = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return words

def build_dataset(words, vocabulary_size=50000):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count = unk_count + 1
		data.append(index)
		count[0][1] = unk_count
  	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  	return data, count, dictionary, reverse_dictionary

def generate_cbow_batch(data, data_index, batch_size, context_size):
	batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = context_size + 1
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	middle = context_size / 2
	for i in range(batch_size):
		labels[i] = buffer[middle]
		context = list(buffer)
		del context[middle]
		batch[i]  = context
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels, data_index

class Word2Vec():
	def __init__(self,
		batch_size=128,
		embedding_size=128, # Dimension of the embedding vector.
		context_size=6, 	# How many words to consider left and right.
		valid_size=16, 		# Random set of words to evaluate similarity on.
		valid_window=100,	# Only pick dev samples in the head of the distribution.
		num_sampled=64,		# Number of negative examples to sample.
		vocabulary_size=50000
	):
		self.batch_size = batch_size
		self.embedding_size = embedding_size
		self.context_size = context_size
		self.valid_size = valid_size
		self.valid_window = valid_window
		self.num_sampled = num_sampled
		self.vocabulary_size = vocabulary_size

		self.data_index = 0

		self._init_graph()
		self.sess = tf.Session(graph=self.graph)

	def _init_graph(self):
		self.valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))
		self.graph = tf.Graph()
		with self.graph.as_default():
			# Input data.
			self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.context_size])
			self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
			self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

			# Variables.
			self.embeddings = tf.Variable(
				tf.random_uniform([self.vocabulary_size, self.embedding_size],-1.0, 1.0))
			self.softmax_weights = tf.Variable(
				tf.truncated_normal([self.vocabulary_size, self.embedding_size],
								 stddev=1.0 / math.sqrt(self.embedding_size)))
			self.softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

			# Model.
			# Look up embeddings for inputs.
			embed = tf.zeros([self.batch_size, self.embedding_size])
			for i in range(self.context_size):
				embed += tf.nn.embedding_lookup(self.embeddings, self.train_dataset[:, i])
			self.embed = embed / self.context_size

			# Compute the softmax loss, using a sample of the negative labels each time.
			self.loss = tf.reduce_mean(
			tf.nn.sampled_softmax_loss(self.softmax_weights, self.softmax_biases, self.embed,
									   self.train_labels, self.num_sampled, self.vocabulary_size))
			# Optimizer.
			# Note: The optimizer will optimize the softmax_weights AND the embeddings.
			# This is because the embeddings are defined as a variable quantity and the
			# optimizer's `minimize` method will by default modify all variable quantities
			# that contribute to the tensor it is passed.
			# See docs on `tf.train.Optimizer.minimize()` for more details.
			self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

			# Compute the similarity between minibatch examples and all embeddings.
			# We use the cosine distance:
			norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
			self.normalized_embeddings = self.embeddings / norm
			self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
			self.similarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddings))

			# init op
			self.init_op = tf.global_variables_initializer()

			# create a saver
			self.saver = tf.train.Saver()

	def train(self, words, num_steps=100001):
		self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(words)
		del words
		print('Most common words (+UNK)', self.count[:5])
		print('Sample data', self.data[:10])

		self.sess.run(self.init_op)
		average_loss = 0
		for step in range(num_steps):
			batch_data, batch_labels, new_data_index = generate_cbow_batch(self.data, self.data_index,
													self.batch_size, self.context_size)
			self.data_index = new_data_index
			feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
			_, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
			average_loss += l
			if step % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step %d: %f' % (step, average_loss))
				average_loss = 0
			# note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 2000 == 0:
				sim = self.similarity.eval(session=self.sess)
				for i in range(self.valid_size):
					valid_word = self.reverse_dictionary[self.valid_examples[i]]
					top_k = 8 # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k+1]
					log = 'Nearest to %s:' % valid_word
					for k in range(top_k):
						close_word = self.reverse_dictionary[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)

		self.final_embeddings = self.sess.run(self.normalized_embeddings)
		return self

if __name__ == '__main__':
	main()
