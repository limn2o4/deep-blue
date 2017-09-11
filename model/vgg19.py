import tensorflow as tf
import numpy as np
import scipy.io

#The VGG_19 NETWORK

VGG19_Layers = ('conv1_1','relu1_1','conv1_2','relu1_2','pool1',
                
                'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
                
                'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',

                'conv4_1','relu4_1','conv4_2', 'relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',

                'conv5_1','relu5_1','conv5_2', 'relu5_2','conv5_3','relu5_3','conv5_4','relu5_4'
                )

def build_convLayer(input,weight,bias):
    layer = tf.nn.conv2d(input,tf.constant(weight),strides=(1,1,1,1),padding="SAME")
    return tf.nn.bias_add(layer,bias)


def build_poolingLayer(input,pooling_type):
    if pooling_type == "max":
        return tf.nn.max_pool(input,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")
    else:
        return tf.nn.avg_pool(input,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")


def load_data(path):
    data = scipy.io.loadmat(path)
    mean = np.mean(data['normalization'][0][0][0],axis=(0,1))
    weights = data['layers'][0]
    return weights


def load_net(weights,input_image,pooling_type):
        net = {}
        cur = input_image
        for i,name in enumerate(VGG19_Layers):
            if name[:4] == 'conv':
                weight,bias = weights[i][0][0][0]
                weight = np.transpose(weight,(1,0,2,3))
                bias = np.reshape(-1)
                cur = build_convLayer(cur,weight,bias)
            elif name[:4] == 'relu':
                cur = tf.nn.relu(cur)
            elif name[:4] == 'pool':
                cur = build_poolingLayer(cur,pooling_type)
            net[name] = cur
        assert len(net) == len(VGG19_Layers)
        return net


def preProcess(image,mean):
    return image - mean