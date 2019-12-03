import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Extract MNIST')
parser.add_argument('--dataset', type=str, required=True, help='path to MNIST dataset')
parser.add_argument('--num_train', type=int, default=53600)
args = parser.parse_args()

magics = {
    'labels': 2049,
    'images': 2051
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(dataset, full=False):
    dataset_type = 'labels' if 'labels' in os.path.basename(dataset) else 'images'
    split = 'train' if 'train' in os.path.basename(dataset) else 'test'

    with open(dataset, 'rb') as f:
        magic_number = np.fromstring(f.read(4), dtype=np.int32)
        magic_number = np.ndarray.byteswap(magic_number)
        assert magic_number == magics[dataset_type]
        number_of_items = np.fromstring(f.read(4), dtype=np.int32)
        number_of_items = np.ndarray.byteswap(number_of_items)
        number_of_items = np.asscalar(number_of_items)

        print('found {} items'.format(number_of_items))

        if dataset_type == 'images':
            width  = np.fromstring(f.read(4), dtype=np.int32)
            height = np.fromstring(f.read(4), dtype=np.int32)
            width = np.ndarray.byteswap(width)
            height = np.ndarray.byteswap(height)
            width = np.asscalar(width)
            height = np.asscalar(height)

            assert width == height and height == 28
        else:
            labels = []
        validation_index = 0
        for i in range(number_of_items):
            if full:
                s = 'full_train'
                split = s
            else:
                s = split if i < args.num_train else 'validation'

            if dataset_type == 'labels':
                label = np.fromstring(f.read(1), dtype=np.uint8)
                label = np.asscalar(label)
                labels.append(label)
            else:
                dest = os.path.join(os.path.dirname(dataset), s, dataset_type)
                image = np.fromstring(f.read(height*width), dtype=np.uint8)
                image = np.reshape(image, (height,width,1))
                index = validation_index if s == 'validation' else i
                image_name = str(index).zfill(5)+'.png'
                image = np.array([ 255 - x for x in image])
                cv2.imwrite(os.path.join(dest, image_name), image)   
                if s == 'validation':
                    validation_index += 1
                print('saved image {} in {}'.format(image_name, dest))
        
        if dataset_type == 'labels':
            dest = os.path.dirname(dataset)
            if split == 'test':
                with open(os.path.join(dest,'test','labels.txt'), 'w') as test_file:
                    for label in labels:
                        test_file.write('{}\n'.format(label))
                print('written labels.txt in {}/test'.format(dest))
            elif split == 'full_train':
                with open(os.path.join(dest,'full_train', 'labels.txt'), 'w') as train_file:
                        for i, label in enumerate(labels):
                            train_file.write('{}\n'.format(label))
            else:
                with open(os.path.join(dest,'train', 'labels.txt'), 'w') as train_file:
                    with open(os.path.join(dest,'validation', 'labels.txt'), 'w') as val_file:
                        for i, label in enumerate(labels):
                            if i < args.num_train:
                                train_file.write('{}\n'.format(label))
                            else:
                                val_file.write('{}\n'.format(label))

if __name__ == '__main__':
    looking_for_files = ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 't10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte']
    dest = os.path.dirname(args.dataset)
    create_dir(os.path.join(dest,'test','images'))
    create_dir(os.path.join(dest,'train','images'))
    create_dir(os.path.join(dest,'validation','images'))
    create_dir(os.path.join(dest,'full_train','images'))
    
    for x in looking_for_files:
        extract(os.path.join(args.dataset, x))
    for x in  ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte']:
        extract(os.path.join(args.dataset, x), full=True)
    
    with open(os.path.join(dest,'full_train','labels.txt'),'r') as f:
        full_train_labels = f.readlines()
    with open(os.path.join(dest,'full_train','dataset.txt'),'w') as f:
        for i in range(60000):
            img = str(i).zfill(5)+'.png'
            label = full_train_labels[i]
            f.write('{} {}'.format(img, label))
    print('Done!')