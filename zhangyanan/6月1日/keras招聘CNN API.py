# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月1日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization


#   ---------------------- 数据的载入和处理 ----------------------------
job_detail_pd = pd.read_csv('D:\\Study\\project\\招聘信息文本分类\\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label)


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)


#  -------------------------- 分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)

#  -------------------------- 建立字典 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50  词嵌入前的预处理
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)  # 长度都填充到50
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()  # 把数组转化为列表


# ------------------------函数API-------------------
from keras.layers import Input
from keras.models import Model
inputs = Input(shape=(50,))
x = Embedding(output_dim=32, input_dim=2000)(inputs)
x = Conv1D(256, 3, activation='relu')(x)  # 训练256个卷积核，每个卷积核大小3*3
x = MaxPool1D()(x)
x = Conv1D(32, 3, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)  # 解决数据分布不平衡
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(units=10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)




batch_size = 256
epochs = 5


model.summary()  # 可视化模型
model.compile(loss="sparse_categorical_crossentropy",  # 多分类
              optimizer="adam",
              metrics=["acc"])

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
    # 训练集的20%用作验证集
)
#  -------------------------- 5、训练模型 -------------------------------


#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
from keras.utils import plot_model

# 保存模型
model.save('model_CNN_text.h5')  # 生成模型文件 'my_model.h5'
# 模型可视化
plot_model(model, to_file='model_CNN_text.png', show_shapes=True)

from keras.models import load_model

# 加载模型
# model = load_model('model_CNN_text.h5')
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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

#  -------------------------- 6、保存模型，显示运行结果 -------------------------------