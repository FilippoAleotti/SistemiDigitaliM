import tensorflow as tf
import numpy as np
import argparse
import time
import random
import os
import cv2
import datetime
from utils import prepare_label, read_labels, extract_central_crop
from networks.factory import get_network
import imutils

parser = argparse.ArgumentParser(description='Run a training of the Network')
parser.add_argument('--training_folder', type=str, required=True, help='path to training split of MNIST dataset')
parser.add_argument('--validation_folder', type=str, required=True, help='path to validation split of MNIST dataset')
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--step', type=int, help='training steps', default=18000)
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--validation_step', type=int, help='step for validation', default=1000)
parser.add_argument('--network', type=str, help='selected network', default='lenet')

args = parser.parse_args()

def train():
    global_step = tf.Variable(0, trainable=False)
    x = tf.placeholder(tf.float32,shape=[args.batch,32,32,1], name='input')
    labels = tf.placeholder(tf.float32, shape=[args.batch,10], name='labels')
    network = get_network(args.network)(x=x, labels=labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    loss = network.loss
    optimization_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

    number_of_params = 0
    for variable in tf.trainable_variables():
        number_of_params += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(number_of_params))
    
    training_images = os.path.join(args.training_folder, 'images')
    training_labels = read_labels(os.path.join(args.training_folder,'labels.txt'))

    validation_images = os.path.join(args.validation_folder, 'images')
    validation_labels = read_labels(os.path.join(args.validation_folder,'labels.txt'))

    config = tf.ConfigProto(allow_soft_placement=True)
    start = time.time()
    best_accuracy = 0
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        valid_images = []
        for step in range(args.step):
            next_batch = get_training_batch(training_images, training_labels, batch_size=args.batch)
            loss_value, _ = session.run([loss, optimization_op], feed_dict={x:next_batch['images'], labels:next_batch['labels']})
            elapsed_time = time.time() - start
            if step and step % 1000 == 0:
                print('step:{}/{} | loss:{} | elapsed time:{}h:{}m:{}s'.format(step, args.step, loss_value, elapsed_time // (3600), elapsed_time // (60), int(elapsed_time)))
            
            if step and  args.validation_step > 0 and step % args.validation_step == 0:
                good_predictions = 0
                for validation_step in range(0, len(validation_labels), args.batch):
                    next_validation_batch = get_validation_batch(validation_images, validation_labels, batch_size=args.batch, starting_index=validation_step)
                    predicted_value = session.run(network.prediction, feed_dict={x:next_validation_batch['images'], labels:next_validation_batch['labels']})
                    gts = np.argmax(next_validation_batch['labels'], axis=-1)
                    good_predictions += (predicted_value == gts).sum()
                print('Validation: {}/{} good predictions'.format(good_predictions, len(validation_labels)))
                if good_predictions >= best_accuracy:
                    saver.save(session, './trained_models/best')
                    best_accuracy = good_predictions
        saver.save(session, './trained_models/model', global_step=args.step)
    print('Training ended!')

def get_training_batch(dataset, labels, batch_size):
    ''' Load next training batch '''
    batch = {
        'images' : [],
        'labels': []
    }
    for i in range(batch_size):
        index = random.randint(0, len(labels)-1)
        label = labels[index]
        image = cv2.imread(os.path.join(dataset, str(index).zfill(5)+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.pad(image, ((2,2),(2,2)),'constant')
        image = image / 255.
        h,w = image.shape
        image = np.expand_dims(image, axis=-1)
        label = prepare_label(label)
        batch['images'].append(image)
        batch['labels'].append(label)
    
    # stacking elements in a single tensor
    batch['images'] = np.array(batch['images'])
    batch['labels'] = np.array(batch['labels'])
    return batch

def get_validation_batch(dataset, labels, batch_size, starting_index):
    ''' Load next training batch '''
    batch = {
        'images' : [],
        'labels': []
    }
    for index in range(starting_index, starting_index + batch_size):
        label = labels[index]
        image = cv2.imread(os.path.join(dataset, str(index).zfill(5)+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.pad(image, ((2,2),(2,2)),'constant')
        image = image / 255.
        image = np.expand_dims(image, axis=-1)
        label = prepare_label(label)
        batch['images'].append(image)
        batch['labels'].append(label)
    # stacking elements in a single tensor
    batch['images'] = np.array(batch['images'])
    batch['labels'] = np.array(batch['labels'])

    return batch
    
if __name__ == '__main__':
    train()