# ----------------------开发者信息-----------------------------------------
 # -*- coding: utf-8 -*-
 # @Time: 2020/5/29 20:04
 # @Author: MiJizong
 # @Version: 1.0
 # @FileName: 1.0.py
 # @Software: PyCharm
 # ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 jieba, Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, MaxPool1D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 数据目录:  D:\Office_software\PyCharm\keras_datasets\job_detail_dataset.csv
from numpy import shape

job_detail_pd = pd.read_csv('D:\\Office_software\\PyCharm\\keras_datasets\\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))                            # 打印数据的前五行
label = list(job_detail_pd['PositionType'].unique())    # 提取所有岗位标签（非重复）
print(label)                                            # 打印标签


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)   # 将所有岗位与数字标签相对应
job_detail_pd = job_detail_pd.dropna()                                        # 删除空行
job_detail_pd.head(5)                                                         # 展示前五行数据


#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):               # 使用jieba分词器进行分词
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut) # 执行分词
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=())) # 使用默认的TF-IDF模型对文档进行分析
                                    # 参数withWeight设置为True时可以显示词的权重，topK设置显示的词的个数

job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 -------------------------------

#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现频率排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”  将中文描述转化为数字化标签
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5、训练模型 -------------------------------
batch_size = 256
epochs = 5
inputs = Input(shape=(50,))
x = Embedding(2000, 32)(inputs)# 输入维度、全连接嵌入的维度
x = Conv1D(256, 3, padding='same', activation='relu')(x) # “same”代表保留边界处的卷积结果
x = MaxPool1D(3, 3, padding='same')(x) # 1D输入的最大池化层
x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = Flatten()(x) # 降维 返回一个一维数组
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)        # 全连接层
x = Dropout(0.2)(x)
x = Dense(units=10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()  # 可视化模型
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
#  -------------------------- 5、训练模型 -------------------------------

#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
# 保存模型
model.save('model_CNN_text.h5')  # 生成模型文件

# 模型可视化
plot_model(model, to_file='model_MLP_text.png', show_shapes=True)

# 加载模型
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))    # 输入的为上一次新输出
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])


# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')  # 给图加上图例
plt.savefig('Valid_acc.png')                      # 保存
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
#  -------------------------- 6、保存模型，显示运行结果 -------------------------------