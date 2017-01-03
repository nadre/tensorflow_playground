import collections
import math
import os
import random
import re
import json
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from functools import wraps
from time import time

'''
tensorflow word2vec example
@author: Erdan Genc
@references:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
https://github.com/wangz10/tensorflow-playground/blob/master/word2vec.py
http://stackoverflow.com/questions/41265035/tensorflow-why-there-are-3-files-after-saving-the-model
'''


def main(load_path=None):
  if load_path != None:
    _word2vec = Word2Vec.load(load_path)
    _word2vec.evaluate()
  else:
    filename = maybe_download('text8.zip', 31344016)
    words = read_words(filename)
    print('Data size %d' % len(words))
    _word2vec = Word2Vec(words)
    _word2vec = _word2vec.train()
    _word2vec.save_all()
  return _word2vec

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    elapsed = time() - start
    print("TIME : %s took %d sec to finish" % (f.__name__, elapsed))
    return result
  return wrapper

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

@timed
def dictionary_to_vocab_file(dictionary, file_path):
  print("Writing vocab.")
  with open(file_path, 'w') as f:
    f.writelines('{0}\n'.format(k) for k in dictionary.keys())

@timed
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
  middle = context_size // 2
  for i in range(batch_size):
    labels[i] = buffer[middle]
    context = list(buffer)
    del context[middle]
    batch[i]  = context
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels, data_index


class Word2Vec:
  def __init__(self,
               words,
               num_steps=1000000, 	# Iterate over n batches
               batch_size=128,
               embedding_size=128, # Dimension of the embedding vector.
               context_size=8, 	# How many words to consider left and right.
               context_weights=[1,2,3,4,4,3,2,1], # Weights of the considered words
               valid_size=16, 		# Random set of words to evaluate similarity on.
               valid_window=100,	# Only pick dev samples in the head of the distribution.
               num_sampled=64,		# Number of negative examples to sample.
               vocabulary_size=50000,  # Use the n most common words
               learning_rate=1.0, 	    # Learning rate to start with
               learning_decay_steps=5000,  # Decay after n steps
               learning_decay=0.999,    # Amount of decay 0.95 -> 5% decay
               checkpoint_steps=10000,	# Save model after n steps
               save_path='checkpoints/', # Folder to save data in
               log_path='checkpoints/',	 # Folder to save logs in
               data_index = 0
               ):

    assert context_size == len(context_weights)

    #remove old logs
    if tf.gfile.Exists(log_path):
      print('Deleting old logs in: %s'%(log_path))
      tf.gfile.DeleteRecursively(log_path)
    tf.gfile.MakeDirs(log_path)

    self.num_steps = num_steps
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.context_size = context_size
    self.context_weights = context_weights
    self.valid_size = valid_size
    self.valid_window = valid_window
    self.num_sampled = num_sampled
    self.vocabulary_size = vocabulary_size
    self.learning_rate = learning_rate
    self.learning_decay_steps = learning_decay_steps
    self.learning_decay = learning_decay
    self.checkpoint_steps = checkpoint_steps
    self.save_path = save_path
    self.log_path = log_path

    self.data_index = data_index

    self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(words)
    del words
    print('Most common words (+UNK)', self.count[:5])
    print('Sample data', self.data[:10])

    # This is where tensorboard takes the labels for the embedding visu
    dictionary_to_vocab_file(self.dictionary ,os.path.join(self.log_path, 'dictionary.vocab'))

    self._init_graph()

    # Tells tensorflow not to allocate all the GPU memory there is
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    self.sess = tf.Session(graph=self.graph, config=config)

  def get_params(self):
    params = {
      'num_steps' : self.num_steps,
      'batch_size' : self.batch_size,
      'embedding_size' : self.embedding_size,
      'context_size' : self.context_size,
      'context_weights' : self.context_weights,
      'valid_size' : self.valid_size,
      'valid_window' : self.valid_window,
      'num_sampled' : self.num_sampled,
      'vocabulary_size' : self.vocabulary_size,
      'learning_rate' : self.learning_rate,
      'learning_decay_steps' : self.learning_decay_steps,
      'learning_decay' : self.learning_decay,
      'checkpoint_steps' : self.checkpoint_steps,
      'save_path' : self.save_path,
      'log_path' : self.log_path,
      'data_index' : self.data_index
    }
    return params

  @timed
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

      # Apply weights to the context words in the training data
      # Idea: Give more weight to words that are closer to the center
      self.train_dataset = self.train_dataset * self.context_weights

      # Model.
      # Look up embeddings for inputs.
      embed = tf.zeros([self.batch_size, self.embedding_size])
      for i in range(self.context_size):
        embed += tf.nn.embedding_lookup(self.embeddings, self.train_dataset[:, i])
      # self.embed = embed / self.context_size
      self.embed = tf.nn.l2_normalize(embed, 1)

      # Compute the softmax loss, using a sample of the negative labels each time.
      self.loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(self.softmax_weights, self.softmax_biases,
                                   self.embed, self.train_labels,
                                   self.num_sampled, self.vocabulary_size))
      tf.summary.scalar('loss', self.loss)

      self.global_step = tf.Variable(0, name="global_step")  # count the number of steps taken
      self.tf_learning_rate = tf.train.exponential_decay(
        self.learning_rate, self.global_step,
        self.learning_decay_steps, self.learning_decay,
        staircase=True)
      tf.summary.scalar('learning_rate', self.tf_learning_rate)

      # Optimizer.
      # Note: The optimizer will optimize the softmax_weights AND the embeddings.
      # This is because the embeddings are defined as a variable quantity and the
      # optimizer's `minimize` method will by default modify all variable quantities
      # that contribute to the tensor it is passed.
      # See docs on `tf.train.Optimizer.minimize()` for more details.
      self.optimizer = tf.train.AdagradOptimizer(self.tf_learning_rate).minimize(self.loss, self.global_step)

      # Compute the similarity between minibatch examples and all embeddings.
      # We use the cosine distance:
      norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))

      self.normalized_embeddings = self.embeddings / norm
      self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)

      self.similarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddings))

      # create a saver
      self.saver = tf.train.Saver()

      # create summary writers for tensorboard
      self.summary_op = tf.summary.merge_all()
      self.summary_writer = tf.summary.FileWriter(self.log_path, self.graph)

      # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
      projector_config = projector.ProjectorConfig()
      embedding = projector_config.embeddings.add()
      embedding.tensor_name = self.normalized_embeddings.name
      embedding.metadata_path = os.path.join(self.log_path, 'dictionary.vocab')

      # Saves a projector_configuration file that TensorBoard will read during startup.
      projector.visualize_embeddings(self.summary_writer, projector_config)

      # init op
      self.init_op = tf.global_variables_initializer()

  @timed
  def train(self):
    start_time = time()

    # Save params once at the beginning of training
    self.save_params()

    # Init variables
    self.sess.run(self.init_op)

    average_loss = 0
    for step in range(self.num_steps):
      batch_data, batch_labels, new_data_index = generate_cbow_batch(self.data, self.data_index,
                                                                     self.batch_size, self.context_size)
      self.data_index = new_data_index
      feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
      _, l, summary = self.sess.run([self.optimizer, self.loss, self.summary_op], feed_dict=feed_dict)
      average_loss += l
      if step % 500 == 0:
        if step > 0:
          average_loss = average_loss / 2000
          tf.summary.scalar('average_loss', average_loss)
          self.summary_writer.add_summary(summary, step)
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('## Average loss at step %d: %f' % (step, average_loss))
        #print('Current learningrate: %f' % (tf.unpack(self.tf_learning_rate)))
        average_loss = 0

      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % self.checkpoint_steps == 0:
        self.save_model()
        self.evaluate()
        elapsed_time = (time() - start_time) / 60
        progress = (float(step) / self.num_steps) * 100
        print( "## Elapsed time: %2.2f min; Progress : %3.2f" % (elapsed_time, progress) )


    self.final_embeddings = self.sess.run(self.normalized_embeddings)
    return self

  @timed
  def evaluate(self, top_k=5):
    sim = self.similarity.eval(session=self.sess)
    for i in range(self.valid_size):
      valid_word = self.reverse_dictionary[self.valid_examples[i]]
      nearest = (-sim[i, :]).argsort()[1:top_k+1]
      log = 'Nearest to %s:' % valid_word
      for k in range(top_k):
        close_word = self.reverse_dictionary[nearest[k]]
        log = '%s %s,' % (log, close_word)
      print(log)

  @timed
  def save_model(self, path=None):
    if path == None:
      path = self.save_path
    self.saver.save(self.sess,
                    os.path.join(path, 'model.ckpt'))
    print("Model has been saved")

  #currently not used
  def save_data_and_labels(self, path=None):
    if path == None:
      path = self.save_path
    json.dump(self.data,
              open(os.path.join(path, 'data.json'), 'w'))
    json.dump(self.count,
              open(os.path.join(path, 'count.json'), 'w'))

  def save_params(self, path=None):
    if path == None:
      path = self.save_path
    params = self.get_params()
    json.dump(params,
              open(os.path.join(path, 'model_params.json'), 'w'))
    json.dump(self.dictionary,
              open(os.path.join(path, 'model_dict.json'), 'w'))
    json.dump(self.reverse_dictionary,
              open(os.path.join(path, 'model_rdict.json'), 'w'))
    print("Params and dictionaries have been saved")

  def save_all(self, path=None):
    '''
    To save trained model and its params.
    '''
    if path == None:
      path = self.save_path

    # save the model (tensorflow)
    self.save_model(path)

    # save parameters of the model
    self.save_params(path)

    # save training data and labels
    # self.save_data_and_labels(path)

    print("Everything got saved in : %s" % self.save_path)

  @classmethod
  def load(cls, path):
    '''
    To restore a saved model.
    '''
    # load params of the model
    path_dir = os.path.dirname(path)
    params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'r'))
    # init an instance of this class
    word2vec = Word2Vec(**params)
    word2vec._restore(os.path.join(path_dir, 'model.ckpt'))
    # evaluate the Variable normalized_embeddings and bind to final_embeddings
    word2vec.final_embeddings = word2vec.sess.run(word2vec.normalized_embeddings)
    # bind dictionaries
    word2vec.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'r'))
    reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'r'))
    # convert indices loaded from json back to int since json does not allow int as keys
    word2vec.reverse_dictionary = {int(key):val for key, val in reverse_dictionary.items()}

    return word2vec

  def _restore(self, path):
    with self.graph.as_default():
      meta_path = path+'.meta'
      print('loading model: '+meta_path)
      self.saver = tf.train.import_meta_graph(meta_path)
      self.saver.restore(self.sess, path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='train word2vec network')
  parser.add_argument('-l', help='path to folder to load model from', type=str)
  args = parser.parse_args()
  main(load_path=args.l)
