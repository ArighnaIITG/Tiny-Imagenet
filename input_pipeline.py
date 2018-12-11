import tensorflow as tf
import numpy as np 
import glob, re
import random

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


