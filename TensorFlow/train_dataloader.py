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

class Dataloader(object):
    def __init__(self, data_path, mode='training', batch_size=1):
        self.data_path = data_path
        self.mode = mode
        self.batch_size = batch_size
        self.dataset = self.prepare_dataset()
        if self.mode == 'training':
            self.iterator =self.dataset.make_initializable_iterator()
        else:
            self.iterator = self.dataset.make_one_shot_iterator()

    def prepare_dataset(self):
        with tf.variable_scope('prepare_dataset'):
            dataset = tf.data.TextLineDataset(os.path.join(self.data_path, 'dataset.txt'))
            if self.mode == 'training':
                dataset = dataset.shuffle(buffer_size=60000)
                dataset = dataset.map(self.prepare_batch, num_parallel_calls=8)
                dataset = dataset.repeat()
                dataset = dataset.prefetch(30)
            else:
                dataset = dataset.map(self.prepare_batch)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            return dataset

    def prepare_batch(self, line):
        with tf.variable_scope("Dataloader"):
            split_line = tf.string_split([line]).values
            img   = tf.strings.join([self.data_path, '/images/', split_line[0]])
            label = tf.string_to_number(split_line[1], out_type=tf.int32)
            label = tf.one_hot(label, depth=10)
            label = tf.cast(label, tf.float32)
            img   = self.read_png_image(img)
        return [img, label]
    
    def read_png_image(self, path):
        with tf.variable_scope("read_png_image"):
            image  = tf.image.decode_png(tf.io.read_file(path))
            image  = tf.image.convert_image_dtype(image,  tf.float32)
            image.set_shape([28,28,1])
            image  = tf.pad(image, [[2,2],[2,2],[0,0]], mode='CONSTANT') 
            return image
    
    def get_next_batch(self):
        ''' Return the next batch. Override to show portion of the full batch '''
        with tf.variable_scope('get_next_batch'):
            return self.iterator.get_next()
    
    def initialize(self, session):
        with tf.variable_scope('initialize'):
            session.run(self.iterator.initializer)

def train():
    global_step = tf.Variable(0, trainable=False)
    dataloader = Dataloader(args.training_folder, batch_size=128)
    imgs, labels = dataloader.get_next_batch()
    network = get_network(args.network)(x=imgs, labels=labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    loss = network.loss
    optimization_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

    number_of_params = 0
    for variable in tf.trainable_variables():
        number_of_params += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(number_of_params))
    
    config = tf.ConfigProto(allow_soft_placement=True)
    start = time.time()
    best_accuracy = 0
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        dataloader.initialize(session)
        for step in range(args.step):
            loss_value, _ = session.run([loss, optimization_op])
            elapsed_time = time.time() - start
            if step and step % 1000 == 0:
                print('step:{}/{} | loss:{} | elapsed time:{}h:{}m:{}s'.format(step, args.step, loss_value, elapsed_time // (3600), elapsed_time // (60), int(elapsed_time)))
        saver.save(session, './trained_models/model', global_step=args.step)
    print('Training ended!')

if __name__ == '__main__':
    train()