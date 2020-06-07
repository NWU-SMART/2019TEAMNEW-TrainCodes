# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.1
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
import matplotlib.pyplot as plt

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


# /------------------ 自编码器模型的建立--------------------*/
# 自编码器将输入复制到输出中，使隐藏表示的h拥有有用的属性
# API模型
from keras.layers import Input,Dense
from keras.models import Model
input = Input(shape=(784,))
hidden = Dense(64,activation='relu')(input)
out = Dense(784,activation='softmax')(hidden)
# 序贯模型
# from keras import Sequential
# model = Sequential()
# model.add(Dense(units=64,input_shape=(784,),activation='relu'))
# model.add(Dense(units=784,activation='softmax'))
# 类继承模型
# from keras.layers import Dense
# class model(keras.Model):
#     def __init__(self):
#         super(model,self).__init__(name='autoEncoder')
#         self.hidden = Dense(units=64,input_shape=(x_train.shape[:1],),activation='relu')
#         self.out = Dense(units=784,activation='softmax')


# 定义自编码器的输入和输出
model = Model(inputs=input,outputs=out)
model.compile(optimizer='adam',loss='mse')
# /------------------ 自编码器模型的建立--------------------*/

# /------------------ 模型训练--------------------*/

result = model.fit(x_train,x_train,batch_size=128,
                   epochs=5,validation_data=(x_test,x_test))
# /------------------ 模型训练--------------------*/

# /------------------训练结果显示--------------------*/
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# 模型保存
model.save('第三章自编码器.h5')

# /------------------ 模型训练--------------------*/

# /------------------ 查看效果--------------------*/
# 查看encoder效果
encoder = Model(input,hidden)
#预测结果
img_encoder = encoder.predict(x_test)
# 打印图片显示encoder效果
n = 10
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(img_encoder[i].reshape(8,8))
    plt.gray()#显示灰度图像
plt.show()

# 查看decoder效果
decoder = model.predict(x_test)

# 打印图片显示decoder效果
n = 10
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(decoder[i].reshape(28,28))
    plt.gray()#显示灰度图像
plt.show()
