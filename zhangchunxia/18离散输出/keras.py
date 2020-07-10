# -----------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月10日
# 内容:离散输出，单标签、多分类
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 keras的包
# 2、建立模型，预测输出，模型训练
# ----------------------   代码布局： ------------------------------------
import numpy as np
import pandas as pd
#-------------数据是自己构造的，分有三类，建立模型-------------------
N=100#每一类的数目
D=2 #维度
K=3#类别数目
X=np.zeros((N*K,D))
Y=np.zeros(N*K,dtype='uint8')#类别标签
for j in range(K):
    ix=list(range(N*j,N*(j+1)))
    r=np.linspace(0.0,1,N)#radius
    t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2  # theta
    X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
    Y[ix]=j#打标签
#将y转换为one-shot编码
Y=np.eye(K)[Y]
from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(10,activation='relu',input_shape=(2,)))
model.add(layers.Dense(3,activation='softmax'))
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X,Y,batch_size=50,epochs=1000)
#------------重新生成数据，测试模型------------------
X=np.zeros((N*K,D))
Y=np.zeros(N*K,dtype='uint8')#类别标签
for j in range(K):
    ix = list(range(N * j, N * (j + 1)))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    Y[ix] = j  # 打标签
#将y转换为one-shot编码
Y=np.eye(K)[Y]
#检测模型在测试集上的表现是否良好
test_loss,test_acc=model.evaluate(X,Y)
print('test_acc:',test_acc)