# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/23 002318:13
# 文件名称：文本分类
# 开发工具：PyCharm


import pandas as pd
from keras.models import Sequential
from  keras.layers.embeddings import Embedding
import keras

# 本地数据路径，先运行修改数据文件.py，将文件修改后在运行此代码
x_train_path = "D:\DataList\job\dataset_x_train.csv"
y_train_path = 'D:\DataList\job\dataset_y_train.csv'
x_train = pd.read_csv(x_train_path).values
y_train = pd.read_csv(y_train_path).values.reshape(len(x_train)).tolist()

print(x_train)
print(y_train)

# ------------------------------------创建模型--------------------------------------

global model


def firstModel():
    model = Sequential()  # 初始化
    model.add(Embedding(output_dim=32,input_dim=2000,input_length=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dense(10,activation='softmax'))
    return model

# 打印模型
model = firstModel()
print(model.summary())

# -------------------------------模型编译----------------------------------------
# 属于多分类问题，loss用sparse_categorical_crossentropy，激活函数输出为概率
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
              )

#-------------------------------训练--------------------------------------------
# validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
history = model.fit(x_train,y_train,batch_size=256,epochs=5,verbose=2,validation_split=0.2)

#--------------------------------查看模型准确率----------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['acc']) # 训练集准确率
plt.plot(history.history['val_acc']) # 验证集准确率
plt.legend()
plt.show()
# 画损失
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend()
plt.show()
# plt.savefig('Valid_loss.png') #保存图片



# -------------------------------------保存模型--------------------------------------
from keras.utils import plot_model
model.save('model_MLP_text.h5') # 保存在当前目录
plot_model(model,to_file='model_MLP.text.png',show_shapes=True)
