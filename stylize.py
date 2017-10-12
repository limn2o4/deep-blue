import tensorflow as tf
import numpy as np
import cv2
import scipy
import model.vgg19 as vgg
from functools import reduce
CONTENT_LAYER = ('relu4_2','relu5_2')
STYLE_LAYER = ('relu1_1','relu2_1','relu3_1','relu4_1','relu5_1')


def stylization(network,init,content,style,content_weight,style_weight,tv_weight,layer_exp):

    # init
    vgg_weight,vgg_mean = vgg.load_net(network)

    shape = (1,content.shape())
    style_shape = (1,) + style.shape()
    content_feature = {}
    style_feature = {}

    layer_weight = 1.0
    style_layer_weight = {}
    for layer in STYLE_LAYER:
        style_layer_weight[layer] = layer_weight
        layer_weight *= layer_exp


    #normalization
    layer_init = 1.0
    layer_sum = 0
    layer_weight = {}
    for style_layer in STYLE_LAYER:
        layer_weight[style_layer] = layer_init
        layer_sum += layer_init
        layer_init *= layer_exp
    for style_layer in STYLE_LAYER:
        layer_weight /= layer_sum

    #get features of two images
    graph = tf.Graph()
    with graph.as_default(),graph.device('/cpu:0'),tf.Session() as sess :
        image = tf.placeholder(tf.float32,shape = content_shape)
        input_layer = vgg.load_net(vgg_weight,image,'max')
        content_pre = np.array([vgg.preProcess(content,vgg_mean)])
        for layer in CONTENT_LAYER:
            content_feature[layer] = input_layer[layer].eval(feed_dict={image:content_pre})

    with graph.as_default(),graph.device('/cpu:0'),tf.Session() as sess :
        image = tf.placeholder(tf.float32,shape = style_shape)
        input_layer = vgg.load_net(vgg_weight,image,'max')
        style_pre = np.array([vgg.preProcess(style,vgg_mean)])
        for layer in STYLE_LAYER:
            feature = input_layer[layer].eval(feed_dict={image:style_pre})
            feature = np.reshape(feature,(-1,feature.shape[3]))
            gram = np.matmul(feature.T,feature)/feature.size
            style_feature[layer] = gram
    init_noise_coeff = 1.0 - init

    #make image
    with tf.Graph().as_default():
        noise  = np.random.normal(size=style_shape,scale=np.std(content)*0.1)
        initial = tf.random_normal(style_shape)*0.256

        image = tf.Variable(initial)
        net = vgg.load_net(vgg_weight,image,"max")

        #content loss
        content_layer_weights = {}
        content_layer_weights['relu4_2'] = content_weight
        content_layer_weights['relu5_2'] = 1.0 - content_weight

        conten_loss = 0
        content_losses = []
        for layer in CONTENT_LAYER:
            content_losses.append(content_layer_weights[layer]*content_weight*(2*tf.nn.l2_loss(net[layer]-content_feature[layer])/content_feature[layer].size))
        conten_loss += reduce(tf.add,conten_loss)

        #style loss
        style_loss = 0
        style_losses = []
        for layer in STYLE_LAYER:
            _,height,width,number = map(lambda i:i.value,net[layer].getshape())
            size = height * width * number
            feat = tf.reshape(net[layer],(-1,number))
            gram = tf.matmul(tf.transpose(feat),feat)/size
            style_gram = style_feature[layer]
            style_losses.append(style_weight[layer]*2*tf.nn.l2_loss(gram - style_gram)/style_gram.size())
        style_loss = style_weight * style_weight * reduce(tf.add,style_losses)

        #denoising
        y_size = get_tensor_size(image[:,1:,:,:])
        x_size = get_tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 *(tf.nn.l2_loss(image[:,1:,:,:], - image[:,:shape[1]-1,:,:])/y_size) + \
                  (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2] - 1:])/x_size)

        loss = conten_loss + style_loss + tv_loss

        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)


        best_loss = float('inf')
        best = None
        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(10):
                train_step.run()
                now_loss = loss.eval()
                if now_loss < best_loss:
                    best_loss = now_loss
                    best = image.eval()
            img_out = vgg.unprocess(best.reshpe(shape[1:]),vgg_mean)





def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul,(d.val for d in tensor.get_shape()))
