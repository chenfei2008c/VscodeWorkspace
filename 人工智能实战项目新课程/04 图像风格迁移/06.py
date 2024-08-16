# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io
from PIL import Image
import os
import time

# 打印当前的系统时间，用于记录训练过程的时间戳。便于计算耗时
def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

# 定义内容图像、风格图像、输出路径。
CONTENT_IMG = 'content.jpg'
STYLE_IMG = 'style5.jpg'
OUTPUT_DIR = 'neural_style_transfer_tensorflow/'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# 定义输出图像的宽度、高度和通道数（RGB为3）。
IMAGE_W, IMAGE_H, COLOR_C = 800, 600, 3

# 定义噪声图像在初始输入图像中的权重（随机叠加的噪音层）、内容损失和风格损失函数的因子。
NOISE_RATIO, BETA, ALPHA = 0.7, 5, 100

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

def load_vgg_model(path):
    try:
        vgg = scipy.io.loadmat(path)
    except FileNotFoundError:
        raise Exception(f"VGG model file not found at {path}")
    
    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        layer_data = vgg_layers[0][layer][0][0]
        layer_name = layer_data[0][0]
        assert layer_name == expected_layer_name, f"Expected layer name {expected_layer_name}, but got {layer_name}"
        W = tf.constant(layer_data[2][0][0])
        b = tf.constant(layer_data[2][0][1].reshape(-1))
        return W, b

    def _conv2d_relu(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        return tf.nn.relu(tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
    graph.update({
        'conv1_1': _conv2d_relu(graph['input'], 0, 'conv1_1'),
        'conv1_2': _conv2d_relu(graph['conv1_1'], 2, 'conv1_2'),
        'avgpool1': _avgpool(graph['conv1_2']),
        'conv2_1': _conv2d_relu(graph['avgpool1'], 5, 'conv2_1'),
        'conv2_2': _conv2d_relu(graph['conv2_1'], 7, 'conv2_2'),
        'avgpool2': _avgpool(graph['conv2_2']),
        'conv3_1': _conv2d_relu(graph['avgpool2'], 10, 'conv3_1'),
        'conv3_2': _conv2d_relu(graph['conv3_1'], 12, 'conv3_2'),
        'conv3_3': _conv2d_relu(graph['conv3_2'], 14, 'conv3_3'),
        'conv3_4': _conv2d_relu(graph['conv3_3'], 16, 'conv3_4'),
        'avgpool3': _avgpool(graph['conv3_4']),
        'conv4_1': _conv2d_relu(graph['avgpool3'], 19, 'conv4_1'),
        'conv4_2': _conv2d_relu(graph['conv4_1'], 21, 'conv4_2'),
        'conv4_3': _conv2d_relu(graph['conv4_2'], 23, 'conv4_3'),
        'conv4_4': _conv2d_relu(graph['conv4_3'], 25, 'conv4_4'),
        'avgpool4': _avgpool(graph['conv4_4']),
        'conv5_1': _conv2d_relu(graph['avgpool4'], 28, 'conv5_1'),
        'conv5_2': _conv2d_relu(graph['conv5_1'], 30, 'conv5_2'),
        'conv5_3': _conv2d_relu(graph['conv5_2'], 32, 'conv5_3'),
        'conv5_4': _conv2d_relu(graph['conv5_3'], 34, 'conv5_4'),
        'avgpool5': _avgpool(graph['conv5_4']),
    })
    return graph

def content_loss_func(model):
    def _content_loss(p, x):
        return tf.reduce_mean(tf.square(x - p))
    return _content_loss(model['conv4_2'], model['conv4_2'])

STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.0), ('conv3_1', 1.5), ('conv4_1', 3.0), ('conv5_1', 4.0)]

def style_loss_func(model):
    def _gram_matrix(F, N, M):
        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)

    def _style_loss(a, x):
        N = a.shape[3]
        M = tf.size(a) // N
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        return tf.reduce_mean(tf.square(G - A)) / (4 * (N ** 2) * (M ** 2))

    return sum(_style_loss(model[layer_name], model[layer_name]) * w for layer_name, w in STYLE_LAYERS)

def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def load_image(path):
    try:
        image = Image.open(path)
    except FileNotFoundError:
        raise Exception(f"Image file not found at {path}")
    image = image.resize((IMAGE_W, IMAGE_H))
    image = np.array(image).astype(np.float32)
    return np.reshape(image, (1, IMAGE_H, IMAGE_W, COLOR_C)) - MEAN_VALUES

def save_image(path, image):
    image += MEAN_VALUES
    image = np.clip(image[0], 0, 255).astype('uint8')
    Image.fromarray(image).save(path)

# 主程序
the_current_time()

content_image = load_image(CONTENT_IMG)
style_image = load_image(STYLE_IMG)

model = load_vgg_model(VGG_MODEL)

input_image = tf.Variable(generate_noise_image(content_image))

@tf.function
def train_step(input_image, model):
    with tf.GradientTape() as tape:
        model['input'].assign(input_image)
        content_loss = content_loss_func(model)
        style_loss = style_loss_func(model)
        total_loss = BETA * content_loss + ALPHA * style_loss
    grad = tape.gradient(total_loss, input_image)
    optimizer = tf.optimizers.Adam(learning_rate=2.0)
    optimizer.apply_gradients([(grad, input_image)])
    return total_loss

ITERATIONS = 2000
for i in range(ITERATIONS):
    total_loss = train_step(input_image, model)
    if i % 100 == 0:
        the_current_time()
        print(f'Iteration {i}')
        print('Cost:', total_loss.numpy())
        save_image(os.path.join(OUTPUT_DIR, f'output_{i}.jpg'), input_image.numpy())
