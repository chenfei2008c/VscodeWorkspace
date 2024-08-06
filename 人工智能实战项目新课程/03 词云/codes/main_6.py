# -*- coding: utf-8 -*-

from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import jieba

# 打开文本
text = open('xyj.txt').read()

# 中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])

# 颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
	s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
	print(s)
	return s

# 生成对象
mask = np.array(Image.open("color_mask.png"))
wc = WordCloud(color_func=random_color, mask=mask, font_path='Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# 保存到文件
wc.to_file('wordcloud.png')