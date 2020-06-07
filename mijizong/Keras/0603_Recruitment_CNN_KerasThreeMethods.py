# ----------------------开发者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/6/3
# @Author: MiJizong
# @Content:  招聘信息文本分类CNN Keras 三种方法
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ---------------------------------------

#  -------------------------- 1、导入需要包 ----------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras import Input, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
import keras
#  -------------------------- 1、导入需要包 ------------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\Office_software\PyCharm\keras_datasets\job_detail_dataset.csv
from numpy import shape

job_detail_pd = pd.read_csv('D:\\Office_software\\PyCharm\\keras_datasets\\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['PositionType'].unique())  # 提取非重复标签
print(label)


# 为工作描述设置每一行标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)  # 将所有岗位与数字标签相对应
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)  # 展示前五行数据


#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):  # 使用jieba分词器进行分词
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)  # 执行分词
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
    # 使用默认的TF-IDF模型对文档进行分析  参数withWeight设置为True时可以显示词的权重，topK设置显示的词的个数


job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)  # 执行提取
#  -------------------------- 3、分词和提取关键词 -------------------------------

#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”  将中文描述转化为数字化标签
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5.1、Sequential训练模型 --------------------------
model1 = Sequential()
model1.add(Embedding(output_dim=32,  # 词向量的维度
                    input_dim=2000,  # Size of the vocabulary 字典大小
                    input_length=50  # 每个数字列表的长度
                    )
          )

model1.add(Conv1D(256,                      # filters 卷积核的数目(即输出的维度)
                 3,                         # 卷积核大小
                 padding='same',  # 输出大小等于输入大小除以步长向上取整；padding = “VALID”时，输出大小等于输入大小减去滤波器大小加上1，最后再除以步长
                 activation='relu'))
model1.add(MaxPool1D(3,                     # 表示池化窗口的大小
                    3,                      # 池化操作的移动步幅，即步长
                    padding='same'))
model1.add(Conv1D(32, 3, padding='same', activation='relu'))
model1.add(Flatten())                       # 平展
model1.add(Dropout(0.3))                    # 丢弃30%
model1.add(BatchNormalization())            # (批)规范化层
model1.add(Dense(256,                       # 代表该层的输出维度为256
                activation='relu'))
model1.add(Dropout(0.2))                    # 丢弃20%
model1.add(Dense(units=10,                  # 代表该层的输出维度为10
                activation="softmax"))
#  -------------------------- 5.1、Sequential训练模型 --------------------------


#  -------------------------- 5.2、API训练模型 ----------------------------------
inputs = Input(shape=(50,))
x = Embedding(input_dim=2000, output_dim=32, input_length=50)(inputs)
x = Conv1D(256, 3, padding='same', activation='relu')(x)
x = MaxPool1D()(x)
x = Conv1D(32, 3, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)  # 批标准化
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=10, activation='softmax')(x)
model2 = Model(inputs=inputs,outputs=x)
#  -------------------------- 5.2、API训练模型 ----------------------------------

#  -------------------------- 5.3、Class继承训练模型 ----------------------------
class Recruitment(keras.Model):
    def __init__(self):
        super(Recruitment,self).__init__()
        self.embedding = Embedding(input_dim=2000,output_dim=32,input_length=50)
        self.conv1 = Conv1D(256,3,padding='same',activation='relu') # 滤波器 输出维度 卷积核大小
        self.maxpool = MaxPool1D()
        self.conv2 = Conv1D(32,3,padding='same',activation='relu')
        self.flatten = Flatten()
        self.dropout1 = Dropout(0.3)
        self.batchnormal = BatchNormalization()
        self.dense1 = Dense(256,activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense2 = Dense(10, activation='relu')

    def call(self, inputs):   # 实现函数调用
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batchnormal(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

model3 = Recruitment()   # 实例化
print(model3)
#  -------------------------- 5.3、Class继承训练模型 ----------------------------


#  ----------------------------- 6、模型编译 ------------------------------------
batch_size = 256
epochs = 5

# 单GPU版本
model1.summary()  # 可视化模型
model1.compile(loss="sparse_categorical_crossentropy",  # 多分类
              optimizer="adam",  # 优化器选择adam（收敛最快）
              metrics=["accuracy"])

history = model1.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2  # 指定训练集的20%用作验证集
)
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况

#  ----------------------------- 6、模型编译 ------------------------------------

#  -------------------------- 7、保存模型，显示运行结果 --------------------------
from keras.utils import plot_model

# 保存模型
model1.save('model_CNN_text.h5')  # 生成模型文件 'my_model.h5'
# 模型可视化
plot_model(model1, to_file='model_CNN_text.png', show_shapes=True)  # 模型网络结构图输出

# 加载模型
# model = load_model('model_CNN_text.h5')
print(x_train[0])
y_new = model1.predict(x_train[0].reshape(1, 50))  # 生成预测
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
#  -------------------------- 7、保存模型，显示运行结果 --------------------------
