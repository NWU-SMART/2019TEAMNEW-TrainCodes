# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/25
#  文件名称：招聘信息文本分类.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入招聘文本数据
# 3、分词并提取关键字
# 4、建立并使用字典
# 5、训练模型
# 6、保存模型与模型可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,MaxPool1D,Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import  regularizers#正则化
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import  BatchNormalization
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入招聘文本数据 -------------------------------
#文件存在本地E:\keras_datasets\job_detail_dataset.csv
job_detail_pd=pd.read_csv('E:\keras_datasets\job_detail_dataset.csv',encoding='UTF-8')
print(job_detail_pd.head(10))
#查看职位类型标签
label=list(job_detail_pd['PositionType'].unique())
print(label)

#为Job_Description设置标签的id
def label_dataset(row):
    num_label=label.index(row)#返回label列表对应值的索引
    return num_label
job_detail_pd['label']=job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd=job_detail_pd.dropna()#删除空行
job_detail_pd.head(10)

#  -------------------------- 2、导入招聘文本数据-------------------------------

#  -------------------------- 3、分词并提取关键字 -------------------------------
#中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut']=job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)
#提取关键字
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts,topK=50,withWeight=False,allowPOS=()))
job_detail_pd['Job_Description_key_word']=job_detail_pd.Job_Description.apply(key_word_extract)

#  -------------------------- 3、 分词并提取关键字-------------------------------

#  -------------------------- 4、建立并使用字典-------------------------------
#建立2000个词的字典
token=Tokenizer(num_words=2000)
token.fit_on_sequences(job_detail_pd['Job_Description_key_word'])#按单词出现次数排序，排序前2000的单词会列入词典中

#使用token字典将“文字”转化为“数字列表”
Job_Description_Seq=token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
#截长补短让所有“数字列表”的长度都为50
Job_Description_Seq_Padding=sequence.pad_sequences(Job_Description_Seq,maxlen=50)
x_train=Job_Description_Seq_Padding
y_train=job_detail_pd['label'].tolist()
#  -------------------------- 4、建立并使用字典-------------------------------


#  -------------------------- 5、训练模型-------------------------------
batch_size=256
epochs=5
model=Sequential()
model.add(Embedding(output_dim=32,#词向量的维度
                    input_dim=2000,#字典大小
                    input_length=50#每个数字列表的长度
                    )
          )
model.add(Dropout(0.2))
model.add(Flatten())#平展
model.add(Dense(units=256,
                activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=10,
                activation="softmax"))

print(model.summary())#打印模型
#CPU版本
#编译模型
model.compile(loss="sparse_categorical_crossentropy",#多分类交叉熵损失函数
              optimizer="adam",#优化器
              metrics=["accuracy"])
#训练模型
history=model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_split=0.2#将训练集的20%用作验证集
)
#  -------------------------- 5、训练模型 -------------------------------

#  -------------------------- 6、保存模型与模型可视化 -------------------------------
#保存模型
model.save('model_MLP_text.h5')#生成模型文件'my_model.h5'
#模型可视化
plot_model(model,to_file='model_MLP_text.png',show_shapes=True)

#模型的预测功能 加载模型
model=load_model('model_MLP_text.h5')
print(x_train[0])
y_new=model.predict(x_train[0].reshape(1,50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

#训练过程可视化
#绘制训练和验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

#绘制训练和验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
#  -------------------------- 6、保存模型与模型可视化-------------------------------