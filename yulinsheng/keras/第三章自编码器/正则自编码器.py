# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.4
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 代码布局 --------------------*/
# 1.导入需要的包
# 2.读取图片数据
# 3.图片预处理
# 4.构建模型自编码器
# 5.训练
# 6.训练结果可视化
# 7.查看效果
#
# /------------------ 导入需要的包--------------------*/
import numpy as np
# /------------------ 导入需要的包--------------------*/

# /------------------ 读取数据--------------------*/
# 数据路径
path = 'mnist.npz'
data = np.load(path)
# ['x_test', 'x_train', 'y_train', 'y_test']
# print(data.files)
# 读取数据
x_train = data['x_train']#(60000, 28, 28)
x_test = data['x_test']#(10000, 28, 28)
data.close()
# 归一化操作
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
# /------------------ 读取数据--------------------*/

# /------------------ 正则自编码器模型的建立--------------------*/
from keras.layers import Input,Dense
from keras import regularizers
from keras.models import Model
input_encoder = Input(shape=(784,))
# kernel_regularizer：施加在权重上的正则项
# bias_regularizer：施加在偏置向量上的正则项
# activity_regularizer：施加在输出上的正则项
hidden = Dense(units=64,activation='relu',
               activity_regularizer=regularizers.l1(1e-5))(input_encoder)
decoder = Dense(units=784,activation='sigmoid')(hidden)
model = Model(inputs=input_encoder,outputs=decoder)
model.compile(optimizer='adam',loss='mse')
# /------------------ 正则自编码器模型的建立--------------------*/

# /------------------ 模型训练--------------------*/
epoch = 5
batch_size = 128
result = model.fit(x_train,x_train,epochs=epoch,
                   batch_size=batch_size,validation_data=(x_test,x_test))
# /------------------ 模型训练--------------------*/

# /------------------结果显示--------------------*/
import matplotlib.pyplot as plt
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# /------------------结果显示--------------------*/

# /------------------查看效果--------------------*/
model.save('bianma.h5')
decoder = model.predict(x_test)
# 打印图片显示decoder效果
n = 5
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(decoder[i].reshape(28,28))
    img = plt.subplot(2, 5, i+6)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()#显示灰度图像
plt.show()