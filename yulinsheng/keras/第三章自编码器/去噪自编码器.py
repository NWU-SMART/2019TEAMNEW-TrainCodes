# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.5
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
# /------------------ 代码布局 --------------------*/


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
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# 加入噪声
# 生成噪声，利用正态分布，0.5*均值为0方差为1的正太分布，加入到原始的数据中
x_train_noisy = x_train + 0.5*np.random.normal(loc=0,scale=1,size=x_train.shape)
x_test_noisy = x_test + 0.5*np.random.normal(loc=0,scale=1,size=x_test.shape)
# 将噪声限值在0-1的范围,数组中比1大的就为1，比0小的就为0
x_train_noisy = np.clip(x_train_noisy,0,1)
x_test_noisy = np.clip(x_test_noisy,0,1)
# /------------------ 读取数据--------------------*/

# /------------------ 自编码器模型的建立--------------------*/
from keras.layers import Conv2D,MaxPool2D,UpSampling2D,Input
# 编码
# 输入28*28*1
input = Input(shape=(28,28,1))
#经过卷积层 28*28*8
conv1 = Conv2D(filters=8,kernel_size=3,padding='same',activation='relu')(input)
# 进过pool层 14*14*8
pool1 = MaxPool2D(pool_size=2,padding='same')(conv1)
# 经过卷积 14*14*16
conv2 = Conv2D(filters=16,kernel_size=3,padding='same',activation='relu')(pool1)
# 经过pool层 7*7*16
pool2 = MaxPool2D(pool_size=2,padding='same')(conv2)
# 经过卷积层 7*7*32
conv3 = Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(pool2)
# 经过pool层4*4*32
hidden = MaxPool2D(pool_size=2,padding='same')(conv3)

# 解码
# 卷积层4*4*16
conv4 = Conv2D(filters=16,kernel_size=3,padding='same',activation='relu')(hidden)
# 反向pooling 8*8*16
uppool1 = UpSampling2D(size=(2,2))(conv4)
# 卷积层8*8*8
conv5 = Conv2D(filters=8,kernel_size=3,padding='same',activation='relu')(uppool1)
# 反向pool层16*16*8
upool2 = UpSampling2D(size=(2,2))(conv5)
# 卷积层14*14*8
conv6 = Conv2D(filters=4,kernel_size=3,activation='relu')(upool2)
# 反向pool层28*28*4
upool3 = UpSampling2D(size=(2,2))(conv6)
# 卷积层28*28*1
decoder = Conv2D(filters=1,kernel_size=3,padding='same',activation='sigmoid')(upool3)
from keras.models import Model
# 构建模型的输入和输出
encoder = Model(inputs=input,outputs=decoder)
# 调用优化函数是自适应超参数学习
encoder.compile(optimizer='adadelta',loss='binary_crossentropy')
# /------------------ 自编码器模型的建立--------------------*/

# /------------------ 模型训练--------------------*/
epoch = 3
batch_size = 128
result = encoder.fit(x_train_noisy,x_train,
                     batch_size=batch_size,epochs=epoch,
                     validation_data=(x_test_noisy,x_test))

# /------------------ 模型训练--------------------*/
# /------------------ 训练损失可视化--------------------*/
import matplotlib.pyplot as plt
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
# /------------------ 训练损失可视化--------------------*/

# /------------------图片结果可视化--------------------*/

# 查看decoder效果
img_decoder = encoder.predict(x_test_noisy)

# 打印图片显示decoder效果
n = 5
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(img_decoder[i].reshape(28,28))
    plt.gray()  # 显示灰度图像
    img = plt.subplot(2, 5, i + 6)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()#显示灰度图像
plt.show()
# /------------------图片结果可视化--------------------*/