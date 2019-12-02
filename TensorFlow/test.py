import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from networks.factory import get_network
from utils import read_labels

parser = argparse.ArgumentParser(description='Run a training of the Network')
parser.add_argument('--testing_folder', type=str, required=True, help='path to testing split of MNIST dataset')
parser.add_argument('--model', type=str, required=True, help='path to trained model')
parser.add_argument('--network', type=str, help='selected network', default='lenet')

args = parser.parse_args()

def test():
    x = tf.placeholder(tf.float32,shape=[1,32,32,1], name='input')
    network = get_network(args.network)(x=x, is_training=False)
    
    testing_images = os.path.join(args.testing_folder, 'images')
    testing_labels = read_labels(os.path.join(args.testing_folder,'labels.txt'))
    
    loader = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    correct_predictions = 0
    total_number = len(testing_labels)
    with tf.Session(config=config) as session:
        loader.restore(session, args.model)
        for step in range(total_number):
            next_batch = get_test_batch(testing_images, testing_labels, step)
            predicted_value = session.run(network.prediction, feed_dict={x:next_batch['images']})
            if np.squeeze(predicted_value) == testing_labels[step]:
                correct_predictions += 1
        print('Testing results: {}/{} correct predictions Error rate:{}'.format(correct_predictions, total_number,100.*(total_number - correct_predictions)/ total_number))

def get_test_batch(dataset, labels, index):
    ''' Load next training batch '''
    batch = {
        'images' : None,
    }
    image = cv2.imread(os.path.join(dataset, str(index).zfill(5)+'.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.pad(image, ((2,2),(2,2)),'constant')
    image = image / 255.
    image = np.expand_dims(image, axis=-1)
    batch['images'] = np.expand_dims(image,0)
    return batch

if __name__ == '__main__':
    test()