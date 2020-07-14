# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/13 001321:17
# 文件名称：__init__.py
# 开发工具：PyCharm

'''
CGAN是Conditional Generative Adversarial Nets的缩写，也称为条件生成对抗网络
CGAN一种带条件约束的GAN，在生成模型（D）和判别模型（G）的建模中均引入条件变量y（conditional variable y）。
简单来讲，普通的GAN输入的是一个N维的正态分布随机数，而CGAN会为这个随机数添上标签，
其利用Embedding层将正整数（索引值）转换为固定尺寸的稠密向量，并将这个稠密向量与N维的正态分布随机数相乘，从而获得一个有标签的随机数
'''
import time

import numpy as np
import keras
from keras import Input, Model
from keras.layers import LeakyReLU, BatchNormalization, Dense, Reshape, Flatten, Embedding
from keras.optimizers import Adam

path = 'D:\DataList\mnist\mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype('float32') / 255.
y_train = y_train.reshape(60000, 1)
# ------------------------------生成器---------------------------#
'''
两个输入，一个100维的向量，一个数字，输入几，生成几的图片。
将数字通过embedding和Flatten层转为100维的向量。100维向量（数字）在和输入的100维相乘，得到最终的100维向量
然后最终的100维向量通过model输出图片.
即两个输入，一个输出（图像）
'''


def generate():
    G_input = keras.layers.Input(shape=(100,))  # 输入100维向量

    # -----------------处理输入标签----------------------------#
    label_input = keras.layers.Input(shape=(1,), dtype='int32')  # 输入 标签
    # 经过embedding转为1*100的矩阵,10表示label中最大数加一，即最大为9(0-9)，加一为10，输出维度为100
    L = keras.layers.Embedding(10, 100)(label_input)
    # 展开为100维度的向量
    label = keras.layers.Flatten()(L)
    # 将输入的100维随机向量，与输入的标签（100维）相乘.multiply两个向量相乘--->multiply([[1 2 3],[4 5 6]])=[4 10 18]
    # 这儿是keras.layers.multiply,m为小写。不是numpy中的multiply
    model_input = keras.layers.multiply([G_input, label])
    # -----------------处理输入标签end----------------------------#

    G = keras.layers.Dense(256)(model_input)
    G = keras.layers.LeakyReLU(0.2)(G)
    G = keras.layers.BatchNormalization()(G)
    G = keras.layers.Dense(512)(G)
    G = keras.layers.LeakyReLU(0.2)(G)
    G = keras.layers.BatchNormalization()(G)
    G = keras.layers.Dense(28 * 28, activation='sigmoid')(G)
    G_output = keras.layers.Reshape((28, 28, 1))(G)

    return keras.Model([G_input, label_input], G_output)


# ------------------------------生成器end---------------------------#


# ------------------------------辨别器---------------------------#
'''
一个输入（图像），两个输出（真假，属于几），
输入图片。两个输出，一个输出判断图片的真假，一个输出判断是几（是1还是2还是3）
'''


def discriminator():
    D_input = Input(shape=(28, 28, 1))  # 输入一个图像
    D = Flatten()(D_input)  # 展开
    D = Dense(512, activation='relu')(D)
    D = Dense(512, activation='relu')(D)
    D = keras.layers.Dropout(0.4)(D)
    D = Dense(512, activation='relu')(D)
    D = keras.layers.Dropout(0.4)(D)

    # 两个输出，一个判断输出的真假（二分类），一个判断输出类别向量（多分类）
    validity = Dense(1, activation='sigmoid')(D)  #
    label = Dense(10, activation='softmax')(D)

    return Model(D_input, [validity, label])


# ------------------------------辨别器end---------------------------#


# ------------------------------实例化---------------------------#
generate_model = generate()
discriminator_model = discriminator()
generate_model.summary()
discriminator_model.summary()
# ------------------------------实例化end---------------------------#


# ------------------------------辨别器编译---------------------------#
'''
因为辨别器有两个输出，所以要两个损失函数
categorical_crossentropy主要用于(one-hot encoding)比如：
[1,0,0],
[0,1,0],
[0,0,1]
sparse_categorical_crossentropy用于目标结果是个整数(integer)，如1，2，3.
'''
optimizer = Adam(0.0002)
losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']  # 一个二分类，一个多分类
discriminator_model.compile(loss=losses,
                            optimizer=optimizer,
                            metrics=['acc'])
# ------------------------------辨别器编译end---------------------------#


# ------------------------------构建GAN---------------------------#
'''
GAN将生成器和辨别器组合，也就是两个输入和两个输出
'''
G_input1 = Input(shape=(100,))  # 输入100维向量
G_input2 = Input(shape=(1,))  # 输入要生成的数字
generate_img = generate_model([G_input1, G_input2])  # 生成的图像
G_output1, G_output2 = discriminator_model(generate_img)  # 得到两个输出，一个判别真假，一个判别标签

GAN_model = Model([G_input1, G_input2], [G_output1, G_output2])
# ------------------------------构建GAN-end--------------------------#


# ------------------------------编译GAN---------------------------#
opt = Adam(0.0002)
GAN_model.compile(loss=losses,
                  optimizer=opt
                  )


# ------------------------------编译GAN-end--------------------------#

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# ------------------------------训练GAN和辨别器------------------------#
def train(epochs, batch_size):
    y_img_true = np.ones((batch_size, 1))  # 真实图像为1
    y_img_false = np.zeros((batch_size, 1))  # 假图片为0
    for i in range(epochs):
        print('第{}次训练'.format(i))
        # --------------------------构建辨别器训练集----------------------#
        # 辨别器需要一个输入（图像）和两个输出（真假，是几）
        idx = np.random.randint(0, 60000, batch_size)
        # 真实图像和标签
        X_img_true = x_train[idx]
        Y_label_true = y_train[idx]
        # 假图像和标签---->由生成器生成---->生成器生成需要两个输入，即input_false和label_false
        input_false = np.random.normal(0,1,(batch_size,100)) # batch_size*100的矩阵
        Y_label_false = np.random.randint(0,10,(batch_size,1)) # batch_size*1的矩阵
        # 得到假图像
        X_img_false = generate_model.predict([input_false,Y_label_false])

        # X由真实图像和假图像构成，前batch_size为真，后batch_size为假
        X = np.concatenate([X_img_true,X_img_false])
        # y判断真假，前batch_size为真（1），后batch_size为假（0）
        y = np.concatenate([y_img_true,y_img_false])
        # Y判断是几
        Y = np.concatenate([Y_label_true,Y_label_false])
        # --------------------------构建辨别器训练集end----------------------#


        #---------------------------训练辨别器------------------------------#
        make_trainable(discriminator_model,True)
        discriminator_model.train_on_batch(X,[y,Y])
        #---------------------------训练辨别器end------------------------------#



        #---------------------------训练GAN------------------------------#
        make_trainable(discriminator_model,False)
        GAN_model.train_on_batch([input_false,Y_label_false],[y_img_true,Y_label_false])
        #---------------------------训练GAN-end-----------------------------#

        if i==10000:
            generate_model.save('generate_model_1.h5')
        if i==20000:
            generate_model.save('generate_model_2.h5')
        if i==30000:
            generate_model.save('generate_model_3.h5')
        if i==40000:
            generate_model.save('generate_model_4.h5')



# ------------------------------训练GAN和辨别器end------------------------#
start = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
train(40002,256)
end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
print(end)

# 效果图在该目录下test中，最后代码