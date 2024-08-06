# -*- coding: utf-8 -*-

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

# 打开文本
text = open('xyj.txt').read()

# 中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])

# 生成对象
mask = np.array(Image.open("color_mask.png"))
wc = WordCloud(mask=mask, font_path='Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# 保存到文件
wc.to_file('wordcloud.png')