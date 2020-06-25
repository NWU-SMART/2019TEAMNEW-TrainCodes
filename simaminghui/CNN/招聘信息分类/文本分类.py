# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/25 002515:59
# 文件名称：文本分类
# 开发工具：PyCharm
import pandas

# x中单个向量中的值不超过2000
from keras import models
from keras.layers import Embedding, Conv1D, Flatten, MaxPool1D, Dropout, BatchNormalization, Dense
from keras.utils import plot_model

x_train = pandas.read_csv('D:\DataList\job\dataset_x_train.csv').values
y_train = pandas.read_csv('D:\DataList\job\dataset_y_train.csv').values.reshape(len(x_train)).tolist()

print(x_train, x_train.shape)
print(y_train)

# ------------------------------创建模型-----------------------------------
model = models.Sequential()
model.add(Embedding(input_dim=2000,  # 输入向量中最大数值不超过2000
                    output_dim=32,  # 词向量的维度
                    input_length=50,  # 单个向量的长度
                    ))
model.add(Conv1D(256,  # 训练256个卷积核，过滤器的个数
                 3,  # 卷积核大小
                 padding='same',  # 输出与原始数据具有相同的长度
                 activation='relu'

                 ))
model.add(MaxPool1D(3, 3, padding='same'))  # 池化
model.add(Flatten())
model.add(Dropout(0.2))
model.add(BatchNormalization())  # 批标准化, 和普通的数据标准化类似, 是将分散的数据统一的一种做法, 也是优化神经网络的一种方法
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#  -------------------------- 模型编译和训练------------------------------
model.summary()  # 可视化模型
model.compile(loss="sparse_categorical_crossentropy",  # 多分类
              optimizer='adam',
              metrics=['acc']
              )
history = model.fit(x_train,y_train,batch_size=256,epochs=5,validation_split=0.2)

model.save('model_CNN_text.h5')
plot_model(model,to_file='model_CNN_text.png',show_shapes=True)

# 绘制训练 & 验证的准确率
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoches')
plt.legend()
plt.show()

# 损失
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoches')
plt.legend()
plt.savefig('Valid_loss.png')
plt.show()
