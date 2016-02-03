# coding=utf-8

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Negative Sampling Skip-Gram Model for training contrastive word embeddings

Trains word embeddings on a text collection with focus on the contrast to other collections
about the same topic. Text collections are assumed to be similar and have a large overlap
in vocabulary.

This is a modified version of tensorflow/tensorflow/models/embedding/word2vec_optimized.py
The key modifications lies in sampling not from the target text collection ("training data"),
but from all other text collections ("sample data") in a leave-one-out fashion.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

import tensorflow.python.platform

from six.moves import xrange  # pylint: disable=redefined-builtin

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import cPickle as pkl

from tensorflow.models.embedding import gen_word2vec as word2vec
import codecs

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 5,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 1,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 10,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy('france', 'paris', 'russia') and "
    "model.nearby(['proton', 'elephant', 'maxwell']")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval.")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval.")
flags.DEFINE_string("name", "model","Name to identify the model")
FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.
    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics
    # interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Model identifier
    self.name = FLAGS.name



class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = {}
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()



  def forward(self, examples, labels):
    """Build the graph for the forward pass."""
    opts = self._options

    # Declare all variables we need.
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / opts.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size, opts.emb_dim], -init_width, init_width),
        name="emb")
    self._emb = emb

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [emb_dim].
    sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

    # Global step: scalar, i.e., shape [].
    self.global_step = tf.Variable(0, name="global_step")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    print("Sample words %d %s " % (len(opts.sample_words),opts.sample_words))
    print("Vocab words %d %s " % (len(opts.vocab_words), opts.vocab_words))
    print("Sample counts %d %s " % (len(opts.sample_counts.tolist()),opts.sample_counts.tolist()))
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=len(opts.sample_words),#NEW
        distortion=0.75,
        unigrams=opts.sample_counts.tolist())) #NEW: only sample from other collections

    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = tf.nn.embedding_lookup(emb, examples)

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits

  def nce_loss(self, true_logits, sampled_logits):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    opts = self._options
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        true_logits, tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        sampled_logits, tf.zeros_like(sampled_logits))
    
    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor

  def optimize(self, loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    opts = self._options
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
    self._lr = lr
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self._train = train

  def build_graph(self):
    """Build the graph for the full model."""
    opts = self._options
    # The training data. A text file.
    (words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram(filename=opts.train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)

    ###NEW: read sampling corpus (=all files in same dir as train_data except for training data)
    full_path = os.path.realpath(opts.train_data)
    path, filename = os.path.split(full_path)
    sampling_files = []
    for file in os.listdir(path):
        if file.endswith(".txt") or file.endswith(".tok") and file != filename:
            sampling_files.append(path+"/"+file)
    print("Files for sampling: ", ", ".join(sampling_files))

    #write new file as concat of all sampling files
    sample_data = opts.train_data+".sample"
    sample_train_data = sample_data+".train"
    o = codecs.open(sample_data, "w", "utf8")
    oo = codecs.open(sample_train_data, "w", "utf8")
    for sampling_file in sampling_files:
        f = open(sampling_file,"r")
        t = f.read()
        o.write(t.decode("utf8")+" ") #concat all files
        oo.write(t.decode("utf8")+" ")
        f.close()
    o.close()
    t = codecs.open(opts.train_data, "r", "utf8")
    oo.write(t.read().decode("utf8"))
    t.close()
    oo.close()

    # The sampling data. A text file.
    (words_samples, counts_samples, words_per_epoch_samples, b_epoch_samples, b_words_samples, examples_samples,
     labels_samples) = word2vec.skipgram(filename=sample_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)

    #Sampling plus training data for getting full vocabulary for embeddings
    (words_samples_train, counts_samples_train, words_per_epoch_samples_train, b_epoch_samples_train, b_words_samples_train, examples_samples_train,
     labels_samples_train) = word2vec.skipgram(filename=sample_train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)

    (opts.all_words, opts.all_counts,
     all_words_per_epoch) = self._session.run([words_samples_train, counts_samples_train, words_per_epoch])

    (opts.sample_words, opts.sample_counts,
     sample_words_per_epoch) = self._session.run([words_samples, counts_samples, words_per_epoch])

    #first add sample words
    for s in opts.sample_words:
        last_index = len(self._word2id)
        self._word2id.setdefault(s,last_index)

    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])

    #then add training words
    for v in opts.vocab_words:
        last_index = len(self._word2id)
        self._word2id.setdefault(v,last_index)

    print("Word2id: ", self._word2id)

    opts.vocab_size = len(self._word2id) #NOTE: wc20(train)+wc(sample) != wc20(train+sample) -> therefore use word2id (proper union)
    print("Sample file: ", sample_data)
    print("Data file: ", opts.train_data)


    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    self._examples = examples_samples
    self._labels = labels_samples
    #self._id2word = opts.all_words
    #for i, w in enumerate(self._id2word):
    for (w,i) in self._word2id.iteritems():
      self._id2word[i] = w

    print("id2word: ", self._id2word)

    true_logits, sampled_logits = self.forward(examples_samples, labels_samples)
    loss = self.nce_loss(true_logits, sampled_logits)
    tf.scalar_summary("NCE loss", loss)
    self._loss = loss
    self.optimize(loss)

    # Properly initialize all variables.
    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()



  def build_eval_graph(self):
    """Build the eval graph."""
    # Eval graph

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._emb, 1)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, self._options.vocab_size))
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with codecs.open(os.path.join(opts.save_path, "vocab.txt"), "w", "utf8") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.all_words[i]),
                             opts.all_counts[i]))

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(opts.save_path,
                                            graph_def=self._session.graph_def)
    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time, last_summary_time = initial_words, time.time(), 0
    last_checkpoint_time = 0
    while True:
      time.sleep(opts.statistics_interval)  # Reports our progress once a while.
      (epoch, step, loss, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._loss, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
            (epoch, step, lr, loss, rate), end="")
      sys.stdout.flush()
      if now - last_summary_time > opts.summary_interval:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        last_summary_time = now
      if now - last_checkpoint_time > opts.checkpoint_interval:
        self.saver.save(self._session,
                        opts.save_path + "model",
                        global_step=step.astype(int))
        last_checkpoint_time = now
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

    return epoch


  def nearby(self, words, num=40):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def plot_with_labels(low_dim_embs, labels, filename="plots/.tsne.png"):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)

def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.save_path:
    print("--train_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    model = Word2Vec(opts, session)
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      #model.eval()  # Eval analogies.
    # Perform a final save.
    model.saver.save(session,
                     os.path.join(opts.save_path, opts.name+"."+"model.ckpt"),
                     global_step=model.global_step)
    model.nearby(['Switzerland'])


    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = len(model._id2word)
    final_embeddings = model._emb.eval(session)
    pkl.dump(final_embeddings,open("embeddings/"+opts.name+".emb.pkl","wb"))
    pkl.dump(model._word2id, open("dicts/"+opts.name+".w2i.pkl","wb"))
    pkl.dump(model._id2word, open("dicts/"+opts.name+".i2w.pkl","wb"))
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    print(zip(model._id2word.iteritems(),low_dim_embs))
    labels = [model._id2word[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels,"plots/"+opts.name+".tsne.png")

    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy('france', 'paris', 'russia')
      # [1]: model.nearby(['proton', 'elephant', 'maxwell'])
      _start_shell(locals())



if __name__ == "__main__":
  tf.app.run()
