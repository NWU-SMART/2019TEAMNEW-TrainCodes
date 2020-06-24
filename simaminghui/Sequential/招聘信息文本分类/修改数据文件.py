# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/23
# 文件名称：修改数据文件
# 开发工具：PyCharm


import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from pandas import DataFrame
# ------------------------------------加载数据---------------------------------------------
job_detail_pd = pd.read_csv('D:\DataList\job\job_detail_dataset.csv')  # 本地数据
print(job_detail_pd.head(5))  # 查看前5行
# (50000, 2) 2 Index(['PositionType', 'Job_Description'], dtype='object')
print(job_detail_pd.shape, job_detail_pd.ndim, job_detail_pd.keys())
lable = list(job_detail_pd['PositionType'].unique())  # unique去除重复值


# 有list['项目管理', '移动开发', '后端开发', '前端开发', '测试', '高端技术职位', '硬件开发', 'dba', '运维', '企业软件']
# 无list['项目管理' '移动开发' '后端开发' '前端开发' '测试' '高端技术职位' '硬件开发' 'dba' '运维' '企业软件']


# 为工作描述设置标签的id，如项目管理返回0，移动开发返回1，后端开发返回2，以此类推
def label_dataset(row):
    num_lable = lable.index(row)  # 返回lable列表对应值得、索引
    return num_lable


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空格
print(job_detail_pd.head(5))


#  -------------------------- 分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return ' '.join(jieba.cut(row))
print('开始')
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)

# 提取关键字
def key_word_extract(texts):
    return ' '.join(analyse.extract_tags(texts,topK=50,withWeight=False,allowPOS=()))

job_detail_pd['Job_Description_key_word'] = job_detail_pd['Job_Description'].apply(key_word_extract)
print(job_detail_pd.head(5))
print('结束')
print(job_detail_pd.keys())




# 建立2000个词的字典
token = Tokenizer(num_words=2000)

# 防止过多计算，消耗计算机内存
Job_Description_key_word = job_detail_pd['Job_Description_key_word']
token.fit_on_texts(Job_Description_key_word) # 按照单词出现次数排序，排名2000以内的会列入字典

# 使用token字典将文字转化为数字列表
Job_Description_Seq = token.texts_to_sequences(Job_Description_key_word)

# 让所有数字列表长度都为50，
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq,maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist() # 转为列表

print('训练数据',x_train)
print('训练标签',y_train)
print('训练数据：',x_train.shape,x_train.ndim)


# 将数据保存,以便后面进行训练
x_train =DataFrame(x_train)
y_train = DataFrame(y_train)

# 数据
x_train.to_csv("D:\DataList\job\dataset_x_train.csv",index=0) # index=0表示不要索引
# 标签
y_train.to_csv("D:\DataList\job\dataset_y_train.csv",index=0)



