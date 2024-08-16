# -*- coding: utf-8 -*-
# 旧代码，tensorflow版本过旧，部分用例如session已弃用
import tensorflow as tf
import numpy as np
# scipy.io 和 scipy.misc：用于读取和保存图像数据
import scipy.io
import scipy.misc
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

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# VGG19 预训练模型中使用的图像平均值，用于对输入图像进行预处理。
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)).astype('float32')
'''
vgg = scipy.io.loadmat(VGG_MODEL)

vgg_layers = vgg['layers'] # 获取模型的所有层
print(vgg_layers[0][0][0][0][2][0][1])
'''

'''
创建一个 NumPy 数组，包含了三个通道的均值，这三个通道分别对应于图像的 RGB（红、绿、蓝）三个颜色通道。
	•	123.68, 116.779, 103.939：
	•	123.68 是蓝色通道的均值。
	•	116.779 是绿色通道的均值。
	•	103.939 是红色通道的均值。

这些均值是从 ImageNet 数据集计算得出的平均值，VGG19 网络在训练时使用了该数据集，因此对输入图像进行这样的均值调整是为了与网络训练时的输入保持一致。
	•	.reshape((1, 1, 1, 3))：将该数组重塑为形状 (1, 1, 1, 3)，这样可以在后续的计算中方便地与图像进行广播操作。
	•	1, 1, 1 是在对图像进行操作时加入的维度，以确保每个图像像素都能进行对应的减去均值操作。
	•	3 是颜色通道的数量，确保对 RGB 三个通道分别进行均值调整。
	'''

def load_vgg_model(path):
	'''
	VGG-19的每一个层都有一个int标识
	Details of the VGG19 model:
	- 0 is conv1_1 (3, 3, 3, 64)
	- 1 is relu
	- 2 is conv1_2 (3, 3, 64, 64)
	- 3 is relu    
	- 4 is maxpool
	- 5 is conv2_1 (3, 3, 64, 128)
	- 6 is relu
	- 7 is conv2_2 (3, 3, 128, 128)
	- 8 is relu
	- 9 is maxpool
	- 10 is conv3_1 (3, 3, 128, 256)
	- 11 is relu
	- 12 is conv3_2 (3, 3, 256, 256)
	- 13 is relu
	- 14 is conv3_3 (3, 3, 256, 256)
	- 15 is relu
	- 16 is conv3_4 (3, 3, 256, 256)
	- 17 is relu
	- 18 is maxpool
	- 19 is conv4_1 (3, 3, 256, 512)
	- 20 is relu
	- 21 is conv4_2 (3, 3, 512, 512)
	- 22 is relu
	- 23 is conv4_3 (3, 3, 512, 512)
	- 24 is relu
	- 25 is conv4_4 (3, 3, 512, 512)
	- 26 is relu
	- 27 is maxpool
	- 28 is conv5_1 (3, 3, 512, 512)
	- 29 is relu
	- 30 is conv5_2 (3, 3, 512, 512)
	- 31 is relu
	- 32 is conv5_3 (3, 3, 512, 512)
	- 33 is relu
	- 34 is conv5_4 (3, 3, 512, 512)
	- 35 is relu
	- 36 is maxpool
	- 37 is fullyconnected (7, 7, 512, 4096)
	- 38 is relu
	- 39 is fullyconnected (1, 1, 4096, 4096)
	- 40 is relu
	- 41 is fullyconnected (1, 1, 4096, 1000)
	- 42 is softmax
	'''
	vgg = scipy.io.loadmat(path)
	vgg_layers = vgg['layers'] # 获取模型的所有层
	
	'''
	•	scipy.io.loadmat(path)：loadmat 是 SciPy 库中的一个函数，用于从 .mat 文件中加载 MATLAB 格式的数据。.mat 文件是一种用于存储 MATLAB 数据的格式，可以包含多种类型的数据，包括矩阵、数组和结构体。
	•	参数 path：这是 .mat 文件的路径，文件中存储了预训练的 VGG19 模型的权重和架构。这个文件通常是通过从 ImageNet 数据集中训练得到的，并保存为 MATLAB 格式。
	•	vgg：loadmat 返回一个字典，vgg 是包含整个 .mat 文件数据的字典。这个字典的键对应于 .mat 文件中的变量名，值是对应变量的数据。
	'''
	# 提取指定layer[int]的权重W和偏置b，并验证层名是否正确
	def _weights(layer, expected_layer_name):
		W = vgg_layers[0][layer][0][0][2][0][0]
		b = vgg_layers[0][layer][0][0][2][0][1]
		layer_name = vgg_layers[0][layer][0][0][0][0]
		assert layer_name == expected_layer_name
		return W, b

	def _conv2d_relu(prev_layer, layer, layer_name):
		# 将 NumPy 数组 W 和 b 转换为 TensorFlow 常量：
		# W 是卷积核的权重矩阵，保持原有形状。
		# b 被重塑为一维张量，以便于在后续的运算中与输出特征图相加。
		W, b = _weights(layer, layer_name)
		W = tf.constant(W)
		b = tf.constant(np.reshape(b, (b.size)))
		'''
		tf.nn.conv2d：执行 2D 卷积操作。
			•	prev_layer：输入特征图。
			•	filter=W：卷积核（权重）。
			•	strides=[1, 1, 1, 1]：卷积步幅，1 表示每次移动一个像素。步幅的列表 [1, 1, 1, 1] 中，第一个和最后一个 1 是批处理大小和通道大小，通常保持不变。中间两个1表示特征图行、列的步幅。
			•	padding='SAME'：填充方式为“相同”，输出的特征图与输入的尺寸相同。
		tf.nn.relu：应用 ReLU 激活函数于卷积结果，计算公式为 max(0, x)，使网络具备非线性特征。
		'''
		# 在 2.x版本的tf.nn.conv2d 中，参数 filter 应为 filters。
		return tf.nn.relu(tf.nn.conv2d(prev_layer, filters=W, strides=[1, 1, 1, 1], padding='SAME') + b)
	
	# 函数 _avgpool 用于在 TensorFlow 中实现平均池化层（Average Pooling Layer）。池化层是卷积神经网络（CNN）中的一种下采样操作，用于减小特征图的尺寸、减少参数数量、控制过拟合以及增强模型的鲁棒性。
	# 平均池化（Average Pooling）：对每个池化窗口内的像素值求平均，结果作为输出特征图中的一个像素。
	# 下采样：通过池化操作，特征图的尺寸减少，通常能提取更具鲁棒性的特征，并提高模型的计算效率。
	# 鲁棒性（Robustness）是一个在计算机科学和工程领域中常用的术语，指系统或模型在面对不确定性、变化或噪声时，保持稳定和有效工作的能力。在机器学习和深度学习中，鲁棒性是指模型在数据变化或噪声干扰下仍能做出准确预测的能力。
	def _avgpool(prev_layer):
		return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	# graph：存储 VGG19 网络各层输出的字典。逐层搭建 VGG19 网络，每层使用卷积加 ReLU 或池化操作。
	# 前五行代码解释：
	'''
	1. 初始化网络图
	graph = {}：创建一个空字典 graph，用于存储网络各层的输出张量。

	2. 定义输入层
	graph['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')：
		•	创建一个形状为 (1, IMAGE_H, IMAGE_W, COLOR_C) 的零张量，用作输入图像，初始化为全零。
		•	IMAGE_H、IMAGE_W：输入图像的高度和宽度。
		•	COLOR_C：输入图像的颜色通道数（如 RGB 图像为 3）。
		•	tf.Variable：将输入张量定义为 TensorFlow 变量，以便在训练过程中进行更新。

	3. 添加卷积层和激活函数
	graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')：
		•	使用 _conv2d_relu 函数在输入层上应用第一个卷积层，并紧跟 ReLU 激活函数。
		•	layer=0：表示从 VGG19 模型中提取 conv1_1 层的参数。
		•	layer_name='conv1_1'：验证层名称，确保从正确的层中提取参数。
	graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')：
		•	在第一个卷积层的输出上应用第二个卷积层，并紧跟 ReLU 激活函数。
		•	layer=2：表示从 VGG19 模型中提取 conv1_2 层的参数。
		•	layer_name='conv1_2'：验证层名称。

	4. 添加池化层
	graph['avgpool1'] = _avgpool(graph['conv1_2'])：
		•	在第二个卷积层的输出上应用平均池化操作，进行下采样。
		•	_avgpool：使用平均池化减少特征图的尺寸。
	'''
	graph = {}
	graph['input']    = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
	graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
	graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
	graph['avgpool1'] = _avgpool(graph['conv1_2'])
	graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
	graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
	graph['avgpool2'] = _avgpool(graph['conv2_2'])
	graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
	graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
	graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
	graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
	graph['avgpool3'] = _avgpool(graph['conv3_4'])
	graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
	graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
	graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
	graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
	graph['avgpool4'] = _avgpool(graph['conv4_4'])
	graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
	graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
	graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
	graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
	graph['avgpool5'] = _avgpool(graph['conv5_4'])
	return graph

# 定义内容损失函数（Content Loss Function）：
def content_loss_func(content_features, generated_features):
	'''
	•	p：内容图像在特定层（如 conv4_2）的激活（特征图）。
	•	x：生成图像在相同层上的激活。
	•	N：特征图的通道数，即激活的深度。
	•	M：特征图的大小（即宽度乘以高度）。
	'''
	def _content_loss(p, x):
		N = p.shape[3]
		M = p.shape[1] * p.shape[2]
		# 使用均方误差（MSE）来计算内容损失，即生成图像和内容图像在某层特征图之间的误差平方和
		return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
	return _content_loss(content_features, generated_features) # 获取内容图像在 conv4_2 层的特征图,以及生成图并计算损失。

# STYLE_LAYERS定义了每个层在总风格损失中的贡献程度(权重)，每个层用一个元组表示，其中第一个元素是层的名称，第二个元素是该层的权重。
STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.0), ('conv3_1', 1.5), ('conv4_1', 3.0), ('conv5_1', 4.0)]
'''
旧用例的风格损失计算函数
def style_loss_func(style_features, generated_features):
	
	_gram_matrix：计算特征图的 Gram 矩阵，用于捕捉图像的风格信息。
	•	F：特征图，形状为 [1, height, width, channels]。
	•	N：特征图的通道数。
	•	M：特征图的空间大小，即高度乘以宽度。
	•	Ft = tf.reshape(F, (M, N))：将特征图重塑为二维矩阵，其中每一行代表一个通道的特征。
	•	tf.matmul(tf.transpose(Ft), Ft)：计算 Gram 矩阵，结果是通道与通道之间的内积矩阵，表示图像的风格特征。
	
	def _gram_matrix(F, N, M):
		Ft = tf.reshape(F, (M, N))
		return tf.matmul(tf.transpose(Ft), Ft)
	
	
	_style_loss：计算生成图像和风格图像在特定层的风格损失。
	•	a：风格图像在特定层的特征图
	•	x：生成图像在相同层的特征图
	
	def _style_loss(a, x): # a,x分别为风格图像及生成图像在特定层的特征图
		N = a.shape[3] # 特征图的通道数，表示每个特征图有多少个滤波器输出。
		M = a.shape[1] * a.shape[2] # 特征图的空间大小，即高度乘以宽度，表示每个通道内有多少个像素。
		A = _gram_matrix(a, N, M) # 特征图像的Gram矩阵，即风格特征
		G = _gram_matrix(x, N, M) # 生成图像的...
		# 使用均方误差（MSE）计算 Gram 矩阵之间的差异，并进行标准化。
		return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(G - A, 2))

	return sum([_style_loss(model[layer_name], model[layer_name]) * w for layer_name, w in STYLE_LAYERS])
'''
# 新用例的风格损失函数
def style_loss_func(style_features, generated_features):
    def _gram_matrix(F):
        N = F.shape[3]
        M = F.shape[1] * F.shape[2]
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    A = _gram_matrix(style_features)
    G = _gram_matrix(generated_features)
    N = style_features.shape[3]
    M = style_features.shape[1] * style_features.shape[2]
    return (1 / (4 * (N ** 2) * (M ** 2))) * tf.reduce_sum(tf.square(G - A))

def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return tf.Variable(input_image, dtype=tf.float32, trainable=True)

'''
	初始输入图像：神经风格迁移的优化过程从该初始图像开始，通过反复迭代修改以最小化损失函数（内容损失和风格损失的加权和），最终得到融合了内容和风格特征的生成图像。
	探索解空间：通过引入噪声，优化过程可以探索更广泛的解空间，避免过早陷入局部最优解。噪声的引入有助于生成图像突破内容图像的限制，向目标风格图像的方向进行调整。
	控制生成过程：通过调整 noise_ratio，可以控制生成图像在内容和风格之间的偏好。例如，较低的噪声比例通常使初始图像更贴近内容图像，这可能有助于更快地收敛到内容一致的结果。
'''

'''
由于scipy.misc已被弃用，改用使用PIL.Image实现
def load_image(path):
	image = scipy.misc.imread(path) # 读取图像文件，路径由参数 path 指定。
	image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W)) # 将图像调整为指定大小 (IMAGE_H, IMAGE_W)。
	image = np.reshape(image, ((1, ) + image.shape))  # 将图像重塑为 4D 张量 [1, height, width, channels]，以适应神经网络的输入格式。
	image = image - MEAN_VALUES # 减去训练 VGG 网络时使用的均值（MEAN_VALUES），以进行数据标准化。这是预处理步骤，使图像输入符合网络的训练条件。
	return image

def save_image(path, image):
	image = image + MEAN_VALUES # 加回均值以恢复图像的原始颜色范围，逆转预处理步骤。
	image = image[0] # 移除批量维度，将 4D 张量恢复为 3D 图像数据。
	image = np.clip(image, 0, 255).astype('uint8') # 将像素值限制在 0 到 255 之间，并转换为 uint8 类型，以便保存为标准图像格式。
	scipy.misc.imsave(path, image) # 保存图像到指定路径 path。
'''
def load_image(path):
    image = Image.open(path)
    image = image.resize((IMAGE_W, IMAGE_H))
    image = np.array(image).astype('float32')  # 转换为 float32
    image = np.reshape(image, ((1,) + image.shape))
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

the_current_time()

def compute_loss(model, content_image, style_image, generated_image):
    model['input'].assign(content_image)
    content_features = model['conv4_2']

    model['input'].assign(style_image)
    style_features = [model[layer] for layer, _ in STYLE_LAYERS]

    model['input'].assign(generated_image)
    generated_features = [model[layer] for layer, _ in STYLE_LAYERS]

    content_loss = content_loss_func(content_features, model['conv4_2'])
    style_loss = sum(style_loss_func(style_features[i], generated_features[i]) * weight 
                     for i, (_, weight) in enumerate(STYLE_LAYERS))

    total_loss = BETA * content_loss + ALPHA * style_loss
    return total_loss

def optimize_image(content_image, style_image, model, iterations=2000):
    generated_image = generate_noise_image(content_image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            total_loss = compute_loss(model, content_image, style_image, generated_image)

        grads = tape.gradient(total_loss, generated_image)
        if grads is None:
            raise ValueError("梯度为空。请检查模型和图形连接。")
        
        optimizer.apply_gradients([(grads, generated_image)])
        
        if i % 100 == 0:
            output_image = generated_image.numpy()
            print(f'Iteration {i}, Cost: {total_loss.numpy()}')
            save_image(os.path.join(OUTPUT_DIR, f'output_{i}.jpg'), output_image)

def main():
    content_image = load_image(CONTENT_IMG)
    style_image = load_image(STYLE_IMG)
    model = load_vgg_model(VGG_MODEL)
    optimize_image(content_image, style_image, model)

if __name__ == "__main__":
    main()