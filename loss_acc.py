import tensorflow as tf 
import numpy as np

def softmax_ce_loss(logits, labels):
	labels = tf.cast(labels, tf.int32)
	ce_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=1.0)
	tf.summary.scalar('loss', ce_loss)


#Can be used if there is a need to improve accuracy
def softmax_ce_smooth_loss(logits, labels):
	labels = tf.cast(labels, tf.int32)
	one = tf.one_hot(labels, 200, dtype=tf.int32)
	ce_loss = tf.losses.softmax_cross_entropy(one, logits, label_smoothing=0.1)
	tf.summary.scalar('loss', ce_loss)


def accuracy(logits, labels):
	labels = tf.cast(labels, tf.int64)
	pred = tf.argmax(logits, axis=1)
	acc = tf.contrib.metrics.accuracy(pred, labels)
	tf.summary.scalar('acc', acc)
	return acc
