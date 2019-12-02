import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from networks.factory import get_network
from utils import read_labels

parser = argparse.ArgumentParser(description='Run a training of the Network')
parser.add_argument('--img', type=str, required=True, help='path to your image')
parser.add_argument('--model', type=str, required=True, help='path to trained model')
parser.add_argument('--network', type=str, help='selected network', default='lenet')

args = parser.parse_args()

def test_single():
    x = tf.placeholder(tf.float32,shape=[1,32,32,1], name='input')
    network = get_network(args.network)(x=x, is_training=False)

    loader = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as session:
        loader.restore(session, args.model)

        img = cv2.imread(args.img)
        img = prefiltering(img)

        predicted_value = session.run(network.prediction, feed_dict={x:img})
        print(predicted_value)

def prefiltering(img):
    ''' Applying pre-processing image filtering '''
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32,32))
    img = img /255.
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

if __name__ == '__main__':
    test_single()