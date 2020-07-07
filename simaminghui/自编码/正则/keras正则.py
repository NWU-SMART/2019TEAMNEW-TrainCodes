# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/3 000318:48
# 文件名称：keras正则
# 开发工具：PyCharm
'''
正则自编码能防止过拟合
'''
import numpy
from keras import Input, regularizers, Model
from keras.layers import Dense

f = numpy.load('D:\DataList\mnist\mnist.npz')
x_train, x_test = f['x_train'], f['x_test']
f.close()

# 归一化
x_train = x_train.astype('float') / 255.
x_test = x_test.astype('float') / 255.
# 将图像（矩阵）转为向量
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# -------------------------------------构建正则自编码--------------------------
input_size = 784
hidden_size = 32
output_size = 784

x = Input(shape=(784,))
'''
l1正则：||w||。具有稀疏性
l1正则：||w||²
'''
h = Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
r = Dense(784, activation='sigmoid')(h)

model = Model(x, r)
# 编译
model.compile(
    optimizer='adam',
    loss='mse'
)

# 训练
history = model.fit(
    x_train, x_train,
    batch_size=128,
    epochs=15,
    verbose=1,
    validation_data=(x_test, x_test)

)

# 查看效果
import matplotlib.pyplot as plt
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20,6))
for i in range(10):
    # 原图
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 模型预测图
    ax = plt.subplot(2,n,i+11)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 查看损失图
history = history.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper right')
plt.show()