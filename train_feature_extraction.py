import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle
import os, json
import time

nb_classes = 43
learning_rate = 1e-3


# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
	data = pickle.load(f)
y = data['labels']
X = data['features']

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
def evaluate(X_data, y_data, BATCH_SIZE, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    g = tf.get_default_graph()
    with g.as_default():
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def train_model(X_, y_, X_val, y_val,
                learning_rates =[], batch_sizes = [],
                saving_path='.', model='model', EPOCHS=10, n_classes=43, max_to_keep=10):
	X_train = X_.copy()
	y_train = y_.copy()
	if not os.path.exists(saving_path):
		os.mkdir(saving_path)
    
	for learning_rate in learning_rates:
		for batch_size in batch_sizes:
			train_stats = []
			model_name = '{}_{}_{}'.format(model, learning_rate, batch_size)
			if not os.path.exists(os.path.join(saving_path, model_name)):
			    os.mkdir(os.path.join(saving_path, model_name))
			    
			tf.reset_default_graph()
			g = tf.Graph()
            
			with g.as_default():
				x = tf.placeholder(tf.float32, (None, 32, 32, 3))
				resized = tf.image.resize_images(x, (227, 227))
				y = tf.placeholder(tf.int32, (None))
				one_hot_y = tf.one_hot(y, n_classes)

				fc7 = AlexNet(resized, feature_extract=True)
				fc7 = tf.stop_gradient(fc7)

				shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
				fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
				fc8b = tf.Variable(tf.zeros(nb_classes))
				logits = tf.matmul(fc7, fc8W) + fc8b
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
				loss_operation = tf.reduce_mean(cross_entropy)
				optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
				training_operation = optimizer.minimize(loss_operation)

				correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
				accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

				saver = tf.train.Saver(max_to_keep=max_to_keep)

				with tf.Session() as sess:
					sess.run(tf.global_variables_initializer())
					num_examples = len(X_train)
					print("Training...{}".format(model_name))
					print()
					t= time.time()
					for i in range(EPOCHS):
						X_train, y_train = shuffle(X_train, y_train)
						t = time.time()

						for offset in range(0, num_examples, batch_size):
							end = offset + batch_size
							batch_x, batch_y = X_train[offset:end], y_train[offset:end]
							sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

						validation_accuracy = evaluate(X_val, y_val, batch_size, accuracy_operation, x, y)
						train_stats.append([i, validation_accuracy])
						print("    EPOCH {} ...".format(i+1))
						print("    Validation Accuracy = {:.3f}".format(validation_accuracy))
						print()
						saver.save(sess, os.path.join(saving_path, model_name, 'model'), global_step=i+1)
					with open(os.path.join(saving_path, model_name, 'train_stats.json'), 'w') as f:
						json.dump(train_stats, f)
					print('Training time: {}'.format(time.time()-t))
					print('------------------------\n')


# TODO: Train and evaluate the feature extraction model.
train_model(X_train, y_train, X_test, y_test, saving_path='./train', learning_rates=[0.001], batch_sizes=[64])
