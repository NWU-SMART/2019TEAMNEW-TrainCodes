# ----------------开发者信息-----------------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月2日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------
# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.layers import BatchNormalization
#  -------------------------- 1、导入需要包 ---------------------------------
#  -------------------------- 2、招聘数据数据导入 ---------------------------
job_detail_pd = pd.read_csv('D:/northwest/小组视频/3招聘文本分类CNN/job_detail_dataset.csv',encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['Positiontype']).unique #标签
print(label)
#为工作描述设置上标签
def label_dataset(row):
    num_label = label.index(row)
    return num_label
job_detail_pd['label'] = job_detail_pd['Position'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna #删除空行
job_detail_pd.head(5)
#  -------------------------- 2、招聘数据数据导入 ---------------------------
#  -------------------------- 3、分词和提取关键词 ---------------------------
#中文分词
def chinese_word_cut(row):
    return" ".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)
#提取关键字
def key_word_extract(row):
    return" ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
job_detail_pd['Job_Description_key_wprd'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 ---------------------------
#  -------------------------- 4、建立字典，并使用 ---------------------------
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_wprd'])#按单词出现次数排序，排序前2000的单词会列入词典中
Job_Description_Seq = token.text_to_sequences(job_detail_pd['Job_Description_key_wprd'])#将文字转换为数字列表
Job_Description_Seq_Padding = sequence.pad.sequences(Job_Description_Seq,maxlen=50)#截长补短，让所有数字列表长度均为50
x_train = Job_Description_Seq
y_train = job_detail_pd['label'].tolist#数组转列表
#  -------------------------- 4、建立字典，并使用 ----------------------------
#  -------------------------- 5、训练模型 -----------------------------------
#/--------------------------method1-API-------------------------------------
inputs = input(shape=(784,))
#层的实例是可以调用的，他以一个张量为参数，并且输出一个张量
x = Embedding(output_dim=32,intput_dim=2000,input_length=50)(inputs)
x = Conv1D(256,3,padding='same',activation='relu')(x)
x = MaxPool1D(3,3,padding='same')(x)
x = Conv1D(3,3,padding='same',activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
prediction = Dense(10,activation='softmax')(x)
#创建了一个输入层和三个全连接层的模型
model =Model(input=inputs,output=prediction)
model.summary()#可视化模型
model.compile(loss= "sparse_categorical_crossentropy",optimizer="adam",metrics="accuracy")
history = model.fit( x_train,    y_train,  batch_size=256,  epochs=5,  validation_split = 0.2)#用训练集的20%做验证集
#/--------------------------method1-API-------------------------------------
#/--------------------------method2-Sequential------------------------------
model1=Sequential()
model1.add(Embedding(output_dim=32,intput_dim=2000,input_length=50))#每个数字列表的长度
model1.add(Conv1D(256,3,padding='same',activation='relu'))
model1.add(MaxPool1D(3,3,padding='same'))
model1.add(Conv1D(3,3,padding='same',activation='relu'))
model1.add(Flatten())
model1.add(Dropout(0.3))
model1.add(BatchNormalization())#规范化层
model1.add(Dense(256,activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(10,activation='softmax'))
model1.summary()#可视化模型
model.compile(loss= "sparse_categorical_crossentropy",optimizer="adam",metrics="accuracy")
history = model.fit( x_train,    y_train,  batch_size=256,  epochs=5,  validation_split = 0.2)#用训练集的20%做验证集
#/--------------------------method2-Sequential------------------------------
#/--------------------------method3-class-----------------------------------
class ZP(keras.Model):
   def __init__(self):
       super(ZP, self).__init__(name='CNN')
       self.embedding = keras.layer.Embedding(2000,32,intput_length=50)
       self.conv1 = keras.layer.Conv1D(256,3,padding='same',activation='relu')
       self.maxpool1D = keras.layer.Maxpool1D(3,3,padding='same')
       self.conv2 = keras.layer.Conv1D(3,3,padding='same',activation='relu')
       self.flatten = keras.layer.Flatten()
       self.dropout1 = keras.layer.Droput(0.3)
       self.batchnormal = keras.layer.BatchNormalization()
       self.dense1 = keras.layer.Dense(256,activation='relu')
       self.dropout2 = keras.layer.Droput(0.2)
       self.dense2 = keras.layer.Dense(10,activation='softmax')

   def call(self,x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.maxpool1D(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batchnormal(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x
model1.summary()#可视化模型
model.compile(loss= "sparse_categorical_crossentropy",optimizer="adam",metrics="accuracy")
history = model.fit( x_train,    y_train,  batch_size=256,  epochs=5,  validation_split = 0.2)#用训练集的20%做验证集
#/--------------------------method3-class-----------------------------------
#  -------------------------- 5、训练模型 -----------------------------------
#  -------------------------- 6、保存模型，显示运行结果 ----------------------
model.save('model_CNN_text.h5')  # 保存模型  生成模型文件 'my_model.h5'
plot_model(model, to_file='model_CNN_text.png', show_shapes=True)# 模型可视化
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])# 绘制训练 & 验证的准确率值
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_accuracy.png')
plt.show()
plt.plot(history.history['loss'])# 绘制训练 & 验证的损失值
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
#  -------------------------- 6、保存模型，显示运行结果 -------------------------