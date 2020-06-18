# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月18日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import numpy as np
import jieba
import jieba.analyse as analyse
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
sess = tf.InteractiveSession()
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
job_detail_pd = pd.read_csv('data.csv', encoding='UTF-8') # 导入数据
label = list(job_detail_pd['PositionType'].unique())# 将职位设置为标签，unique函数为去除重复项

def label_dataset(row):
     num_label = label.index(row) #按照label出现次序编号
     return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset) #将编号添加到数据中
job_detail_pd = job_detail_pd.dropna() # 删除空行

def chinese_word_cut(row):
     return " ".join(jieba.cut(row))  #使用jieba将句子分割成词汇
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)# 将分割的词汇添加到数据中

def key_word_extract(texts):
  return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
# (待提取的文本,返回多少权重最大的关键词,是否一并返回关键词权重值，是否筛选)
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract) # 将关键词添加到数据中

token = Tokenizer(num_words = 100) # 设置一个100词的字典
token.fit_on_texts(job_detail_pd['Job_Description_key_word']) #按单词出现次数排序，排序前100的单词会存入字典中
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)# 所有“数字列表”长度都是50  

x_train = Job_Description_Seq_Padding    #数组形式，每句话固定50个词，这50个词经过编码，变成了50个数字（0-100），不够50个数字的补上0
y_train = job_detail_pd['label'].tolist()# 标签转换为列表形式
y_train = np.array(y_train)  # 标签转换为array形式
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 3、搭建模型 -------------------------------
in_units = 50  # 输入节点
h1_units = 256  # 隐层输出节点
h2_units = 10  # 输出节点

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  # 隐含层的权重，初始化为截断的正态分布，其标准差为0.1
b1 = tf.Variable(tf.zeros([h1_units]))  # 隐含层的偏置，全部赋值为0

W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))  # 隐含层的权重，初始化为截断的正态分布，其标准差为0.1
b2 = tf.Variable(tf.zeros([h2_units]))  # 隐含层的偏置，全部赋值为0

W3 = tf.Variable(tf.zeros([h2_units, 10]))  # 输出层权重
b3 = tf.Variable(tf.zeros([10]))  # 输出层的偏置

x = tf.placeholder(tf.float32, [None, in_units])  # 定义输入x的placeholder
keep_prob = tf.placeholder(tf.float32)  # #dropout的比率(即保留节点的概率)

hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)  # 定义一个激活函数为ReLU的隐含层hidden1
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  # 实现dropout功能，即随机将一部分节点置为0.其中keep_prob即为保留数据而不置为0的比例

hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2)+b2)  # 定义一个激活函数为ReLU的隐含层hidden1
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)  # 实现dropout功能，即随机将一部分节点置为0.其中keep_prob即为保留数据而不置为0的比例

y = tf.nn.softmax(tf.matmul(hidden2_drop, W3)+b3)  # 得到输出y
y_ = tf.placeholder(tf.float32, [h2_units])  # 定义输出y的placeholder
#  -------------------------- 3、搭建模型 -------------------------------

#  -------------------------- 4、训练模型   --------------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))  # 交叉熵损失函数

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)  # 选择自适应优化器Adagrad，把学习速率设为0.3
tf.global_variables_initializer().run()

for i in range(10):
    batch_xs, batch_ys = x_train, y_train
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    print(y_)
#  -------------------------- 4、训练模型   --------------------------------
