'''
Factory for networks
'''

import tensorflow as tf
from networks import lenet
from collections import namedtuple

NETWORK_FACTORY= {
    'lenet': lenet.Network,
}

def get_network(name):
    AVAILABLE_NETWORKS = NETWORK_FACTORY.keys()
    assert(name in AVAILABLE_NETWORKS)
    return NETWORK_FACTORY[name]