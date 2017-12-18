#reconstruction
from PIL import Image
import scipy.io as io
import tensorflow as tf
import numpy as np
import cv2
import time


STYLE_WEIGHT = 1
CONTENT_WEIGHT = 1
STYLE_LAYER = ['relu1_2','relu2_2','relu3_2']
CONTENT_LAYER = ['relu1_2']

vgg_prarms = None
def load_vgg_params():
    global vgg_prarms
    if vgg_prarms == None:
        vgg_prarms = io.loadmat("D:\project\deep-blue\model\imagenet-vgg-verydeep-19.mat")
    return vgg_prarms
def init_vgg19(input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    load_vgg_params()
    weight = vgg_prarms['layers'][0]

    net = input_image
    network = {}

    for i,name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kel,bias = weight[i][0][0][0][0]
            kel = np.transpose(kel,(1,0,2,3))
            conv = tf.nn.conv2d(net,tf.constant(kel),strides=[1,1,1,1],padding='SAME',name = name)
            net = tf.nn.bias_add(conv,bias.reshape(-1))
            net = tf.nn.relu(net)
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
        network[name] = net
    return network

def content_loss(target_feature,content_feature):
    _,height,width,channel = map(lambda i : i.value,content_feature.get_shape())
    print("content_feature: {}".format(content_feature.get_shape()))

    content_size = height*width*channel
    return tf.nn.l2_loss(target_feature - content_feature)/content_size


def style_loss(target_feature,style_feature):
    _,height,width,channel = map(lambda  i : i.value,target_feature.get_shape())
    print("target_feature : {}".format(target_feature.get_shape()))
    size = height*width*channel
    target_feature = tf.reshape(target_feature,(-1,channel))
    target_gram = tf.matmul(tf.transpose(target_feature),target_feature)/size

    style_feature = tf.reshape(style_feature,(-1,channel))
    style_gram = tf.matmul(tf.transpose(style_feature),style_feature)/size

    return tf.nn.l2_loss(target_gram - style_gram) / size

def total_loss(style_image,content_image,target_image):
    style_feature = init_vgg19([style_image])
    content_feature = init_vgg19([content_image])
    target_feature = init_vgg19([target_image])
    loss = 0.0
    for layer in CONTENT_LAYER:
        loss += CONTENT_WEIGHT * content_loss(target_feature[layer],content_feature[layer])

    for layer in STYLE_LAYER:
        loss += STYLE_WEIGHT * style_loss(target_feature[layer],style_feature[layer])

    return loss

def stylize(style_image,content_image,learning_rate = 0.01,epochs = 100):

    target = tf.get_variable("target",shape = content_image.shape,dtype=tf.float32,initializer=tf.random_normal_initializer)

    style_input = tf.get_variable("style_input",shape=style_image.shape,dtype=tf.float32,initializer=tf.constant_initializer(style_image))

    content_input = tf.get_variable("content_input",shape=content_image.shape,dtype=tf.float32,initializer=tf.constant_initializer(content_image))

    loss = total_loss(style_input,content_input,target)

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess :
        with tf.device('/gpu:0'):
            st = time.time()
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                _,_loss,result_image = sess.run([trainer,loss,target])
                print("step = {} , loss = {}".format(i,_loss))
                if (i +1) % 100 == 0:
                    image = np.clip(result_image+128,0,255).astype(np.uint8)
                    #cv2.imwrite("./nerual{}.jpg".format(i+1),image)
                    Image.fromarray(image).save("./nerual{}.jpg".format(i+1))
            en = time.time()
            print("using :{}".format(st - en))

if __name__ == "__main__":
    # style = cv2.imread("./1.jpg")
    # content = cv2.imread("./lena.jpg")
    style = Image.open("./1.jpg")
    content = Image.open("./lena.jpg")

    style = np.array(style).astype(np.float32) - 128.0
    content = np.array(content).astype(np.float32) - 128.0
    print(style.shape)
    stylize(style,content,0.5,200)

