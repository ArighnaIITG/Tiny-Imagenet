import tensorflow as tf
import numpy as np 

def conv_2D(inputs, filters, kernel_size, name=None):
	"""3*3 conv layer, with stride 1, and same padding, and ReLU non-linear activation"""

	sd = np.sqrt(2/np.prod(kernel_size) * int(inputs.shape[3]))  #For He Normalization
	out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding='same', activation=tf.nn.relu, 
				kernel_initializer=tf.random_normal_initializer(stddev=sd), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), name=name)
	tf.summary.histogram('act_'+ name, out)
	return out

def dense_relu(inputs, units, name=None):
	sd = np.sqrt(2/int(inputs.shape[1]))
	out = tf.layers.dense(inputs, units, activation=tf.nn.relu, 
				kernel_initializer=tf.random_normal_initializer(stddev=sd), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), name=name)
	tf.summary.histogram('act_'+ name, out)
	return out

def dense(inputs, units, name=None):
	sd = np.sqrt(2/int(inputs.shape[1]))
	out = tf.layers.dense(inputs, units,  
				kernel_initializer=tf.random_normal_initializer(stddev=sd), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), name=name)
	tf.summary.histogram('act_'+ name, out)
	return out

def vgg_16(training_batch, config):
	"""Main conv net VGG-16 architecture"""

	img = tf.cast(training_batch, tf.float32)
	out = (img-128.0)/128.0
	tf.summary.histogram('img', training_batch)

	#(N, 56, 56, 3)
	out = conv_2D(out, 64, (3, 3), 'conv1_1')
	out = conv_2D(out, 64, (3, 3), 'conv1_2')
	out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

	# (N, 28, 28, 64)
  	out = conv_2D(out, 128, (3, 3), 'conv2_1')
 	out = conv_2D(out, 128, (3, 3), 'conv2_2')
  	out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

  	# (N, 14, 14, 128)
  	out = conv_2D(out, 256, (3, 3), 'conv3_1')
  	out = conv_2D(out, 256, (3, 3), 'conv3_2')
 	out = conv_2D(out, 256, (3, 3), 'conv3_3')
  	out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool3')

  	# (N, 7, 7, 256)
  	out = conv_2D(out, 512, (3, 3), 'conv4_1')
  	out = conv_2D(out, 512, (3, 3), 'conv4_2')
  	out = conv_2D(out, 512, (3, 3), 'conv4_3')

  	# (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
  	out = tf.contrib.layers.flatten(out)
  	out = dense_relu(out, 4096, 'fc1')
  	out = tf.nn.dropout(out, config.dropout_keep_prob)

  	# (N, 4096) -> (N, 2048)
  	out = dense_relu(out, 2048, 'fc2')
  	out = tf.nn.dropout(out, config.dropout_keep_prob)

  	# softmax
  	# (N, 2048) -> (N, 200)
  	logits = dense(out, 200, 'fc3')

	return logits
