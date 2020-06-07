#-------------------------------开发者信息----------------------------------
#开发人：王园园
#日期：2020.5.25
#开发软件：pycharm
#项目：招聘信息文本分类（keras）

#--------------------------------导入包-------------------------------------
from msilib import sequence
from sre_parse import Tokenizer
from keras import Sequential, Input, Model
from keras.engine.saving import load_model
from keras.layers import Embedding, Conv1D, MaxPool1D, Flatten, Dropout, BatchNormalization, Dense
from keras.utils import plot_model
from networkx.drawing.tests.test_pylab import plt
from numpy import shape
from tensorflow import keras
import pandas as pd

#---------------------------------招聘数据导入--------------------------------


job_details = pd.read_csv('',enconding='UTF-8')            #数据
label = list(job_details['PositionType'].unique())         #标签

#为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)                              #返回label列表对应值的索引
    return num_label

job_details['label']= job_details['PositionType'].apply(label_dataset)
job_details = job_details.dropna()  #删除空行

#---------------------------------分词和提取关键词------------------------------
#中文分词
def chinese_word_cut(row):
    return ' '.join('jieba'.cut(row))

job_details['Job_Description_jieba_cut'] = job_details.Job_Description.apply(chinese_word_cut)

#提取关键词
def key_word_extract(texts):
    return ' '.join('analyse'.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))

job_details['Job_Description-_key_word'] = job_details.Job_Description.apply(key_word_extract)

#------------------------------------建立字典，并使用--------------------------------------
#建立2000个词的字典
token = Tokenizer(num_words = 2000)
#按单词出现次数排序，排序前2000个单词会列入词典中
token.fit_on_texts(job_details['Job_Description-_key_word'])
#使用token字典将‘文字’转化为‘数字列表’
Job_Description_Seq = token.texts_to_sequences(job_details['Job_Description-_key_word'])
#截长补短让所有‘数字列表’长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding      #训练数据
y_train = job_details['lable'].tolist()    #训练标签

#-----------------------------------------构建模型-----------------------------------------
#----------------------------------------Sequential()类型---------------------------------
model1 = Sequential()
#Embedding将离散变量转变为连续向量的方式
model1.add(Embedding(output_dim=32,        #词向量的维度
                     input_dim=2000,       #字典大小
                     input_length=50))     #每个数字列表的长度
model1.add(Conv1D(256,                     #输出大小
                  3,                       #卷积核大小
                  padding='same',
                  activation='relu'))
model1.add(MaxPool1D(3, 3, padding='same')) #最大池化
model1.add(Conv1D(32, 3, padding='same', activation='relu'))
model1.add(Flatten())
model1.add(Dropout(0.3))
model1.add(BatchNormalization())
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(units=10, activation='softmax'))

#-------------------------------------------API类型----------------------------------------
input = Input(shape())
x = Embedding(output_dim=32, input_dim=2000, input_length=50)(input)
x = Conv1D(256, 3, padding='same', activation='relu')(x)
x = MaxPool1D(3, 3, padding='same')(x)
x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=10, activation='softmax')
model2 = Model(inputs=input, outputs=x)

#--------------------------------------------类继承---------------------------------------
input2 = Input(shape())
class Recruitment(keras.Model):
    def __init__(self):
        super(Recruitment, self).__init__(name='MLP')

        self.embedding = keras.layers.Embedding(output_dim=32, input_dim=2000, input_length=50)
        self.conv1 = keras.layers.Conv1D(256, 3, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv1D(32, 3, padding='same', activation='relu')
        self.maxPool1 = keras.layers.MaxPool1D(3, 3, padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(units=10, activation='softmax')
        self.dropout1 = keras.layers.Dropout(0.3)
        self.dropout2 = keras.layers.Dropout(0.2)
        self.batch = keras.layers.BatchNormalization()

    def call(self, input2):
        x = self.embedding(input2)
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batch(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)

model3 = Recruitment()

#---------------------------------------------训练模型--------------------------------
batch_size = 256    #批大小
epochs = 5          #训练5次
model1.summary()    #模型输出
model1.compile(loss='sparse_categorical_crossentropy',    #模型编译
               optimizer='adam',
               metrics=['accuracy'])
#从训练集中抽取0.2进行验证
history = model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#-----------------------------------------------保存模型，可视化--------------------------
#保存模型
model1.save('model_CNN_text.h5')
#模型可视化
plot_model(model1, to_file='model_CNN_text.png', show_shape=True)
#加载模型
model = load_model('model_CNN_text.h5')
y_new = model.predict(x_train[0].reshape(1, 50))
#训练结果可视化
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()




