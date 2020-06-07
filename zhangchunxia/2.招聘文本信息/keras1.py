# ----------------开发者信息------------------------------------------------------
# 开发者：张春霞
# 开发日期：2020年5月27日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------------

# ----------------------   代码布局： --------------------------------------------
# 1、导入程序所需 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： --------------------------------------------

#  -------------------------- 1、导入需要包 --------------------------------------
import panda as pd5#数据处理
import jieba      #中文词库
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer #将文本处理成索引类型的数据
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,MaxPool1D,Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers, Input
import matplotlib.pyplot as plt              #画图
import numpy as np
from keras.utils import plot_model
from numpy import shape
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization
#  -------------------------- 1、导入需要包 --------------------------------------

#  -------------------------- 2、招聘数据数据导入 --------------------------------
from tensorflow_core.python.estimator import keras

from renmei.Keras图像分类 import batch_size

job_detail_pd = pd.read_csv('D:/northwest\小组视频/2招聘文本信息MLP/job_detail_dataset.csv')#数据目录
print(job_detail_pd.head(5))#读前五行
label = list(job_detail_pd['Position'].unique())#职位标签
print(label)
#为工作描述设置标签的ID
def label_dataset(row):
    num_label = label.index(row) #返回label列表对应值的索引
    return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset())#不同的工作类型打标签
job_detail_pd = job_detail_pd.dropna()#删除多余的空行，删除丢失的部分
job_detail_pd.head(5)#打印前五行
#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
#中文分词 ，将我爱吃苹果分为：我/爱/吃/苹果/
def chinese_word_cut(row):
    return"".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Descriptin.apply(chinese_word_cut)#apply函数的返回值就是chinese_word_cut函数的返回值
job_detail_pd.head(5)
#提取关键字
def key_word_extract(texts):
    return "".join(analyse.extract_tags(texts,topk=50,),withWeight=False,allowPOS=())#提取50个
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Descriptin.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 ------------------------------------

#  -------------------------- 4、建立字典，并使用字典 --------------------------------
#建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts('job_Description_key_word')#排序是按单词出现的次序进行的，排序前2000的单词会列入词典中
#使用token字典将“文字“转换为”数字列表“
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
#词嵌入前的预处理，让所有词的长度均为50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq,maxlen=50)
#选取训练集
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist() #数组转为列表
print('*******')
#  -------------------------- 4、建立字典，并使用 -----------------------------------

#  -------------------------- 5、模型训练--------------------------------------------
#-----------------------------sequential方法-----------------------------------------
model = Sequential()
model.add(Embedding(input_dim=1000,output_dim=32,input_length=50))#输入维度是1000，输出维度是32，数字列表是50,batch=50*32
model.add(Dropout(0.2))
model.add(Flatten())#平展batch
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10,activation='softmax'))
print(model.summary())#打印模型
model.compile(loss= "sparse_categorical_crossentropy",  #计算标签和预测值之间的交叉熵损失
              optimizer = "adam"  , #优化器是adam
              metrics = ["acc"]#评估指标
             )
history = model.fit(x_train,y_train,batch_size = 256,epochs =15,verbose = 2,validation_split = 0.2 )#取训练集的20%做验证集
#-----------------------------sequential方法-----------------------------------------
#-------------------------------API方法------------------------------------------------
inputs = Input(shape())
x = Embedding(input_dim=2000,output_dim=32,input_length=50)(input)
x = Conv1D(256,3,padding='same',activation='relu')(x)#一维卷积，输入和输出图像大小相同
x = MaxPool1D(3,3,padding='same')(x)
x = Conv1D(32,3,padding='same',activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = BatchNormalization(x)
x = Dense(256,activation='relu')
x = Dense(10,activation='softmax')
model1 = model(input=inputs,output=x)
model.compile(loss= "sparse_categorical_crossentropy",  #计算标签和预测值之间的交叉熵损失
              optimizer = "adam"  , #优化器是adam
              metrics = ["acc"])#评估指标
model1.fit = (x_train,y_train)
#-------------------------------API方法------------------------------------------------
#-----------------------------class方法------------------------------------------------
class classification(keras.Model):
    def __init__(self):
        super(classification,self).__init__(name='')
        self.embedding = keras.layers,Embedding(input_dim=2000,output_dim=32,input_length=50)
        self.conv1 = keras.layers.Conv1D(256,3,padding='same',activation='relu')
        self.conv2 = keras.layers.Conv1D(32,3,padding='same',activation='relu')
        self.maxPool1 = keras.layers.MaxPool1D(3,3,padding='same')
        self.flattern = keras.layers.Flattern()
        self.dense1 = keras.layers.Dense(256,activation='relu')
        self.dense2 = keras.layers.Dense(10, activation='softmax')
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dropout2 = keras.layers.Dropout(0.25)
        self.batch = keras.layers.BatchNormalization()
    def call(self,inputs):
        x = self.embedding(inputs)
        x = self.conv1
        x = self.maxpool1
        x= self.conv2
        x = self.flattern
        x = self.dropout1
        x = self.batch
        x = self.dense1
        x = self.dropout2
        x = self.dense2
        model = classification()
        model.compile(loss= "sparse_categorical_crossentropy",  #计算标签和预测值之间的交叉熵损失
              optimizer = "adam"  , #优化器是adam
              metrics = ["acc"])#评估指标
        model1.fit = (x_train, y_train)
# -----------------------------class方法----------------------------------------------------
#  -------------------------- 5、模型训练---------------------------------------------------

#  -------------------------- 6、保存模型，显示运行结果 ------------------------------------
from keras.utils import plot_model
model.save('model_MLP_text.h5')#模型保存
plot_model(model,to_file='model_MLP_text.h5',show_shapes=True)
print(x_train[0])#打印训练集第一行
y_new = model.predict(x_train[0].reshape(1,50))
print(y_new)#打印通过softmaxde得到的每个类的概率
print(list(y_new[0].index(max(y_new[0]))))#softmax得到的最大概率值的索引作为预测值输出
print(y_train[0])#输出真实标签
#画准确率曲线
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Test']),
loc='upper left'
plt.savfig('Test_acc.png')
plt.show()
#绘制训练验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['Train','Test']),
loc='upper left'
plt.savfig('Test_loss.png')
plt.show()
#  -------------------------- 6、保存模型，显示运行结果 ------------------------------------