# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/11 001114:31
# 文件名称：test
# 开发工具：PyCharm

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

generator_model = load_model('generator_model_1.5.h5')


# ---------------生成器生成的图像-----------
x = np.random.uniform(0, 1, size=[10, 100]) # 10*100的矩阵
y_img = generator_model.predict(x)
plt.figure(figsize=(20,6))
for i in range(10):
    # 生成器生成图像
    ax = plt.subplot(2,5,i+1)
    plt.imshow(y_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.savefig('generator.png')
plt.show()
