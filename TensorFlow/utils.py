import numpy as np

def read_labels(labels_file):
    with open(labels_file,'r') as f:
        lines = f.readlines()
    lines = [float(l.strip()) for l in lines]
    return lines

def prepare_label(label):
    ''' Given a single scalar label, get a One Hot Vector with 10 values
        For instance, if the label is 5, this method returns the vector
        [0,0,0,0,0,1,0,0,0,0]
    '''
    one_hot = []
    for i in range(int(label)):
        one_hot.append(0.)
    one_hot.append(1.)
    for i in range(int(label)+1, 10):
        one_hot.append(0.)
    return np.array(one_hot)

def extract_central_crop(image, final_w, final_h):
    '''
        Get central crop of image
    '''
    start_h = (image.shape[0] - final_h) // 2
    end_h = start_h + final_h 
    start_w = (image.shape[1] - final_w) // 2
    end_w = start_w + final_w
    return image[start_h:end_h, start_w:end_w]