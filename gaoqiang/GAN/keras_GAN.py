# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.18
# 开发框架：keras
# 代码功能：GAN 生成sin曲线
# 温馨提示：
# ----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、随机生成正弦函数曲线
# 2、定义构建生成器模型函数
# 3、定义构建判别器模型函数
# 4、训练
# 5、分析结果
# ----------------------------------------------------------------------------------------------------------------------#
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import pandas as pd
import numpy as np
# 随机生成正弦函数曲线
def sample_data(n_samples=10000, x_vals=np.arange(0, 5, .1), max_offset=100, mul_range=[1, 2]):
    '''
    x_vals=np.arange(0, 5, .1)  arange函数用于创建等差数组 从0到5，步长0.1
    mul_range=[1, 2]   数组
    '''
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset                                # 生成0-100的随机数
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0]) # 1+（2-1）* 0-1的随意数
        vectors.append(
            np.sin(offset + x_vals * mul) / 2 + .5                              # 构成sin函数
        )
    return np.array(vectors)

# 定义生成器模型
def get_generative(G_in):
    x = Dense(200,activation='tanh')(G_in)
    G_out = Dense(50,activation='tanh')(x)

    G = Model(G_in,G_out)
    opt = SGD(lr=1e-3)
    G.compile(loss='binary_crossentropy',optimizer=opt)
    return G,G_out

G_in = Input(shape=[10])
G, G_out = get_generative(G_in)
G.summary()


# 定义判别器模型
def get_discriminative(D_in):
     x = Reshape((-1,1))(D_in)
     x = Conv1D(50,5,activation='relu')(x) # 输出通道50，核大小5
     x = Dropout(0.25)(x)
     x = Flatten()(x)                      # 拉平，做全连接
     x = Dense(50)(x)
     D_out = Dense(2,activation='sigmoid')(x)

     D = Model(D_in,D_out)
     dopt = Adam(lr=1e-3)
     D.compile(loss='binary_crossentropy',optimizer=dopt)
     return  D,D_out

D_in = Input(shape=[50])
D, D_out = get_discriminative(D_in)
D.summary()

# 串联两个模型
# 首先定义一个函数，目的是每次训练 generator 时要冻住 discriminator
def set_trainability(model,trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
def make_gan(GAN_in,G,D):
    set_trainability(D,False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in,GAN_out)
    GAN.compile(loss='binary_crossentropy',optimizer=G.optimizer)
    return GAN,GAN_out

GAN_in = Input([10])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()

# 现在生成一些假的和真实的数据，并在开始gan之前对鉴别器进行预训练。
def sample_data_and_gen(G, noise_dim=10, n_samples=10000):
    XT = sample_data(n_samples=n_samples)                                # 真实数据XT
    XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])      # 噪声数据(均匀分布[low,high)中随机采样，注意定义域是左闭右开)
    XN = G.predict(XN_noise)                                             # 将噪声数据输入到生成网络中
    X = np.concatenate((XT, XN))                                         # 拼接数据
    y = np.zeros((2*n_samples, 2))                                       # 20000行，2列  都变成0
    y[:n_samples, 1] = 1                                                 # 0-10000的数 第二列 用1替换
    y[n_samples:, 0] = 1                                                 # 10000-20000的数 第一列 用1替换
    return X, y

# 预训练
def pretrain(G, D, noise_dim=10, n_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
    set_trainability(D, True)                                           # 对判别器训练
    D.fit(X, y, epochs=1, batch_size=batch_size)

pretrain(G, D)
# 20000/20000 [==============================] - 6s 312us/step - loss: 0.0087



# 开始训练
# 交替训练 discriminator 和 chained GAN，在训练 chained GAN 时要冻住 discriminator 的参数：
def sample_noise(G, noise_dim=10, n_samples=10000):
    X = np.random.uniform(0, 1, size=[n_samples, noise_dim]) # 噪声数据(均匀分布[low,high)中随机采样，注意定义域是左闭右开)
    y = np.zeros((n_samples, 2))                             # 10000行，2列  都变成0
    y[:, 1] = 1                                              # 0-10000的数 第二列 用1替换
    return X, y


from tqdm import tqdm
# tqdm是Python中专门用于进度条美化的模块，通过在非while的循环体内嵌入tqdm，可以得到一个能更好展现程序运行过程的提示进度条，

def train(GAN,G,D,epochs=500,n_samples=10000,noise_dim=10,batch_size=32,verbose=False):
    d_loss = []                                      # 判别器损失列表
    g_loss = []                                      # 生成器器损失列表
    e_range = range(epochs)                          # 500个epoch
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:
        X,y = sample_data_and_gen(G,n_samples=n_samples,noise_dim=noise_dim)
        set_trainability(D,True)                   # 训练判别器
        d_loss.append(D.train_on_batch(X,y))       # 添加损失

        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)                 # 训练生成器
        g_loss.append(GAN.train_on_batch(X,y))     # 添加损失
        if verbose and (epoch + 1) % 50 == 0:      # 每50个epoch打印一次损失
            print("Epoch #{}:Generative Loss:{},Discriminative Loss:{}".format(epoch + 1,g_loss[-1],d_loss[-1]))
    return d_loss,g_loss

d_loss, g_loss = train(GAN, G, D,verbose=True)

# 结果：
# 10%|█         | 50/500 [02:11<19:52,  2.65s/it]Epoch #50:Generative Loss:3.489975690841675,Discriminative Loss:0.32874369621276855
#  20%|██        | 100/500 [04:17<16:32,  2.48s/it]Epoch #100:Generative Loss:4.226477146148682,Discriminative Loss:0.14711035788059235
#  30%|███       | 150/500 [06:22<14:24,  2.47s/it]Epoch #150:Generative Loss:6.064790725708008,Discriminative Loss:0.08977949619293213
#  40%|████      | 200/500 [08:30<14:38,  2.93s/it]Epoch #200:Generative Loss:4.602912902832031,Discriminative Loss:0.06655096262693405
#  50%|█████     | 250/500 [11:05<11:40,  2.80s/it]Epoch #250:Generative Loss:4.067621231079102,Discriminative Loss:0.047072142362594604
#  60%|██████    | 300/500 [13:34<11:22,  3.41s/it]Epoch #300:Generative Loss:4.033910751342773,Discriminative Loss:0.06385380774736404
#  70%|███████   | 350/500 [16:00<06:22,  2.55s/it]Epoch #350:Generative Loss:4.080207824707031,Discriminative Loss:0.046898555010557175
#  80%|████████  | 400/500 [18:16<04:37,  2.77s/it]Epoch #400:Generative Loss:3.7600913047790527,Discriminative Loss:0.044745996594429016
#  90%|█████████ | 450/500 [20:33<02:18,  2.77s/it]Epoch #450:Generative Loss:3.947572708129883,Discriminative Loss:0.03289871662855148
# 100%|██████████| 500/500 [22:59<00:00,  2.76s/it]
# Epoch #500:Generative Loss:4.041848659515381,Discriminative Loss:0.02629685401916504
# 分析：
# 可以看到相互博弈的过程














