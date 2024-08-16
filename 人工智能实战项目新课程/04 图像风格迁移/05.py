# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image

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
IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

# 定义噪声图像在初始输入图像中的权重（随机叠加的噪音层）、内容损失和风格损失函数的因子。
NOISE_RATIO = 0.7
BETA = 5
ALPHA = 100

# VGG19 预训练模型中使用的图像平均值，用于对输入图像进行预处理。
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)).astype('float32')

# 加载 VGG19 模型，去掉全连接层部分
def load_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = {layer.name: layer.output for layer in vgg.layers}
    return tf.keras.Model([vgg.input], outputs)

# 定义内容损失函数（Content Loss Function）
def content_loss_func(content_features, generated_features):
    return tf.reduce_mean(tf.square(content_features - generated_features))

# 定义风格损失函数（Style Loss Function）
def style_loss_func(style_features, generated_features):
    def _gram_matrix(F):
        gram = tf.linalg.einsum('bijc,bijd->bcd', F, F)
        return gram / tf.cast(F.shape[1] * F.shape[2], tf.float32)

    A = _gram_matrix(style_features)
    G = _gram_matrix(generated_features)
    return tf.reduce_mean(tf.square(A - G))

def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = tf.random.uniform(content_image.shape, -20, 20)
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return tf.Variable(input_image, dtype=tf.float32, trainable=True)

def load_image(path):
    image = Image.open(path)
    image = image.resize((IMAGE_W, IMAGE_H))
    image = np.array(image).astype('float32')  # 转换为 float32
    image = np.expand_dims(image, axis=0)  # 添加批量维度
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

def compute_loss(model, content_image, style_image, generated_image):
    content_features = model(content_image)['block4_conv2']
    style_features = [model(style_image)[name] for name in STYLE_LAYERS]
    generated_features = [model(generated_image)[name] for name in STYLE_LAYERS]

    content_loss = content_loss_func(content_features, model(generated_image)['block4_conv2'])
    style_loss = sum(style_loss_func(style_features[i], generated_features[i]) * weight 
                     for i, (_, weight) in enumerate(STYLE_LAYERS))

    total_loss = BETA * content_loss + ALPHA * style_loss
    return total_loss

STYLE_LAYERS = [
    ('block1_conv1', 0.5), 
    ('block2_conv1', 1.0), 
    ('block3_conv1', 1.5), 
    ('block4_conv1', 3.0), 
    ('block5_conv1', 4.0)
]

def optimize_image(content_image, style_image, model, iterations=2000):
    generated_image = generate_noise_image(content_image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            total_loss = compute_loss(model, content_image, style_image, generated_image)

        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        
        if i % 100 == 0:
            output_image = generated_image.numpy()
            print(f'Iteration {i}, Cost: {total_loss.numpy()}')
            save_image(os.path.join(OUTPUT_DIR, f'output_{i}.jpg'), output_image)

def main():
    content_image = load_image(CONTENT_IMG)
    style_image = load_image(STYLE_IMG)
    model = load_vgg_model()
    optimize_image(content_image, style_image, model)

if __name__ == "__main__":
    main()