import numpy as np
import os
import shutil
import glob
import tensorflow as tf
from datetime import datetime


def build_label_dicts():
	label_dict = {}
	class_desc = {}
	with open('../tiny-imagenet-200/wnids.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			ss = line[:-1]
			label_dict[ss] = i
	with open('../tiny-imagenet-200/words.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			ss, desc = line.split('\t')
			desc = desc[:-1]
			if ss in label_dict:
				class_desc[label_dict[ss]] = desc

	return label_dict, class_desc 


def load_file_labels(mode):
	label_dict, class_desc = build_label_dicts()
	file_labels = []
	if mode == 'train':
		filenames = glob.glob('../tiny-imagenet-200/train/*/images/*.JPEG')
		for fn in filenames:
			match = re.search(r'n\d+', fn)
			label = str(label_dict[match.group()])
			file_labels.append((fn, label))
	elif mode == 'val':
		with open('../tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
			for line in f.readlines():
				split_line = line.split('\t')
				fn = '../tiny-imagenet-200/val/images/' + split_line[0]
				label = str(label_dict[split_line[1]])
				file_labels.append((fn, label))

	return file_labels


def read_image(fq, mode):
	item = fq.dequeue()
	fn = item[0]
	label = item[1]
	file = tf.read_file(fn)
	img = tf.image.decode_jpeg(file, channels=3)
	if mode == 'train':
		img = tf.random_crop(img, np.array([56, 56, 3]))
		img = tf.image.random_flip_left_right(img)
		img = tf.image.random_hue(img, 0.05)
		img = tf.image.random_saturation(img, 0.5, 2.0)
	else:
		img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56)

	label = tf.string_to_number(label, tf.int32)
	label = tf.cast(label, tf.uint8)

	return [img, label]


def batchq(mode, config):
	file_labels = load_file_labels(mode)
	random.shuffle(file_labels)
	fq = tf.train.input_producer(file_labels, num_epochs=config.num_epochs, shuffle=True)
	return tf.train.batch_join([read_image(fq, mode) for i in range(2)], config.batch_size, shapes=[(56, 56, 3), ()],
			capacity=2048)

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


class TrainConfig(object):
	batch_size = 64
	num_epochs = 56
	sum_int = 250
	eval_int = 2000  #integer multiple of eval_interval
	lr= 0.01
	reg = 5e-4
	momentum = 0.9
	dropout_kp = 0.5
	model_name = 'vgg_16'
	model = staticmethod(globals()[model_name]) 


class TrainControl(object):
	"""Basic training control
	   Track Validation accuracy, decreasr lr by 1/5th when val. accuracy worsens.
	"""

	def __init__(self, lr):
		self.val_accs = []
		self.lr = lr
		self.num_lr_updates = 0
		self.lr_factor = 1/5

	def add_val_acc(self, loss):
		self.val_accs.append(loss)

	def update_lr(self, sess):
		if len(self.val_accs) < 3:
			return
		dec = False
		if self.val_accs[-1] < max(self.val_accs):
			dec = True
		avg2 = (self.val_accs[-2] + self.val_accs[-3])/2
		if abs(self.val_accs[-1] - avg2) < 0.002:
			dec = True

		if dec:
			old_lr = sess.run(self.lr)
			self.lr.load(old_lr*self.lr_factor)
			self.num_lr_updates += 1
			print('***New Learning Rate: {}'.format(old_lr*self.lr_factor))

	def done(self):
    	if self.num_lr_updates > 3:
      		return True
    	else:
			return False


def optimizer(loss, config):
	lr = tf.Variable(config.lr, trainable=False, dtype=tf.float32)
	global_step = tf.Variable(0, trainable=False, name='global_step')
	optim = tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True)
	tr_op = optim.minimize(loss, global_step=global_step)

	return tr_op, global_step, lr


# Check this function afterwards.
def get_logdir():
  	"""Return unique logdir based on datetime"""
  	now = datetime.utcnow().strftime("%m%d%H%M%S")
  	logdir = "run-{}/".format(now)

	return logdir


def model(mode, config):
	with tf.device('/cpu:0'):
		imgs, labels = batchq(mode, config)

  	logits = config.model(imgs, config)
  	softmax_ce_loss(logits, labels)
  	acc = accuracy(logits, labels)
  	total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
  	total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                        name='total_loss') * config.reg
  	
  	"""
  	for l2 in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    	# add l2 loss histograms to TensorBoard and cleanup var names
    	name = 'l2_loss_' + l2.name.split('/')[0]
    	tf.summary.histogram(name, l2)
    """

 	return total_loss, acc




def evaluate(ckpt):
  	"""Load checkpoint and run on validation set"""
  	config = TrainConfig()
  	config.dropout_kp = 1.0  # disable dropout for validation
  	config.num_epochs = 1
  	accs, losses = [], []

  	with tf.Graph().as_default():
    	loss, acc = model('val', config)
    	saver = tf.train.Saver()
    	init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    	with tf.Session() as sess:
      		init.run()
      		saver.restore(sess, ckpt)
      		coord = tf.train.Coordinator()
      		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      		try:
        		while not coord.should_stop():
          			step_loss, step_acc = sess.run([loss, acc])
          			accs.append(step_acc)
          			losses.append(step_loss)
      		except tf.errors.OutOfRangeError as e:
        		coord.request_stop(e)
      		finally:
        		coord.request_stop()
        		coord.join(threads)
  	mean_loss, mean_acc = np.mean(losses), np.mean(accs)
  	print('Validation. Loss: {:.3f}, Accuracy: {:.4f}'.
        	format(mean_loss, mean_acc))

	return mean_loss, mean_acc


def options(config):
	config.config_name = 'default'
	ckpt_path = 'checkpoints/' + config.model_name + '/' + config.config_name
	tflog_path = ('tf_logs/' + config.model_name + '/' +
	            config.config_name + '/' + get_logdir())
	checkpoint = None
	if not os.path.isdir(ckpt_path):
		os.makedirs(ckpt_path)
		filenames = glob.glob('*.py')
		for filename in filenames:
			shutil.copy(filename, ckpt_path)
		return False, ckpt_path, tflog_path, checkpoint
	else:
		filenames = glob.glob('*.py')
		for filename in filenames:
			shutil.copy(filename, ckpt_path)
		while True:
			q1 = input('Continue previous training? [Y/n]: ')
			if len(q1) == 0 or q1 == 'n' or q1 == 'Y':
				break
		if q1 == 'n':
			return False, ckpt_path, tflog_path, checkpoint
		else:
			checkpoint = tf.train.latest_checkpoint(ckpt_path)
		return True, ckpt_path, tflog_path, checkpoint

def train():
	config = TrainConfig()
	cont_train, ckpt_path, tfl_path, ckpt = options(config)
	"""
	cont_train = True
	ckpt_path = 'checkpoints/' + config.model_name + '/default'
	tfl_path = ('tf_logs/' + config.model_name + '/' +
			'default' + '/' + get_logdir())
	ckpt = None
	if not os.path.isdir(ckpt_path):
		os.makedirs(ckpt_path)
		fns = glob.glob('*.py')
		for fn in fns:
			shutil.copy(fn, ckpt_path)
		cont_train = False
	else:
		fns = glob.glob('*.py')
		for fn in fns:
			shutil.copy(fn, ckpt_path)
        ckpt = tf.train.latest_checkpoint(ckpt_path)
		cont_train = True
	"""

	g = tf.Graph()
	with g.as_default():
		loss, acc = model('train', config)
		tr_op, g_step, lr = optimizer(loss, config)
		controller = TrainControl(lr)

		val_acc = tf.Variable(0.0, trainable=False)
    	val_loss = tf.Variable(0.0, trainable=False)
    	tf.summary.scalar('val_loss', val_loss)
		tf.summary.scalar('val_accuracy', val_acc)

		init = tf.group(tf.global_variables_initializer(),
			tf.local_variables_initializer())
		[tf.summary.histogram(v.name.replace(':', '_'), v)
				for v in tf.trainable_variables()]
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		summ = tf.summary.merge_all()
    	saver = tf.train.Saver(max_to_keep=1)
		writer = tf.summary.FileWriter(tfl_path, g)

		with tf.Session() as sess:
			init.run()
			if cont_train:
				saver.restore(sess, ckpt)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			try:
				losses, accs = [], []
				while not coord.should_stop():
					step_loss, _, step, step_acc, __ = sess.run([loss, tr_op,
							g_step, acc, extra_update_ops])
					losses.append(step_loss)
					accs.append(step_acc)
					if step%config.eval_interval == 0:
						ckpt = saver.save(sess, ckpt_path + '/model', step)
						ml, macc = evaluate(ckpt)
						val_loss.load(ml)
						val_acc.load(macc)
						controller.add_val_acc(macc)
						controller.update_lr(sess)
						if controller.done():
							break
					if step % config.summary_interval == 0:
            			writer.add_summary(sess.run(summ), step)
            			print('Accuracy: {:.4f}'.
                  				format(100*np.mean(accs)))
						losses, accs = [], []

			except tf.errors.OutOfRangeError as e:
        		coord.request_stop(e)
      		finally:
        		coord.request_stop()
				coord.join(threads)


if __name__ == "__main__":
	train()

