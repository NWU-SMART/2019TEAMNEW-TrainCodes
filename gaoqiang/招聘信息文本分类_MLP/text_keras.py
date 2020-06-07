# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020年5月24日
# 开发框架：keras
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
#--------------------------------------------------------------#
#-----------------------------------------招聘数据导入------------------------------------------------#
import pandas as pd
# 加载数据#
job_detail_pd = pd.read_csv('F:\Keras代码学习\keras\keras_datasets\job_detail_dataset.csv',encoding='utf-8')
print(job_detail_pd.head()) # 打印出前五个
# 十类标签：# ['项目管理', '移动开发', '后端开发', '前端开发', '测试', '高端技术职位', '硬件开发', 'dba', '运维', '企业软件']
label = list(job_detail_pd['PositionType'].unique()) # unique()返回参数数组中所有不同的值，并按照从小到大排序
print(label) # 打印标签

# 返回label列表对应值的索引
def label_dataset(row):
    num_label =label.index(row)
    return num_label

job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset) # 对应起来
job_detail_pd = job_detail_pd.dropna() # 删除空行 （默认滤除所有包含NaN）
print(job_detail_pd.head()) # 打印前五个 (带标签)

#----------------------------------------分词和提取关键词----------------------------------#
import jieba
import jieba.analyse as analyse
'''
jieba库是一款优秀的 Python 第三方中文分词库，jieba 支持三种分词模式：精确模式、全模式和搜索引擎模式。
1.精确模式：试图将语句最精确的切分，不存在冗余数据，适合做文本分析
2.全模式：将语句中所有可能是词的词语都切分出来，速度很快，但是存在冗余数据
3.搜索引擎模式：在精确模式的基础上，对长词再次进行切分

'''
# 中文分词
def chinese_word_cut(row):
    return "".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
print(job_detail_pd.head())# 打印前五个

# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))

job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
print(job_detail_pd.head())# 打印前五个

#---------------------------------------建立字典，并使用-----------------------------------#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
# 建立2000个词的字典
token = Tokenizer(num_words = 2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])#按单词出现次数排序，排序前2000的单词会列入词典中
# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
# 训练数据
x_train = Job_Description_Seq_Padding    # 训练数据 数字列表（长度为50）
y_train = job_detail_pd['label'].tolist()# 训练标签转化为列表

#------------------------------训练模型--------------------------------------#
#------------------------------方法一：序贯模型（从头到尾结构顺序，不分叉）--------------------------------------#
from keras.layers import Dense, Dropout,Activation, Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
'''
Embedding,即嵌入层，将正整数（下标）转换为具有固定大小的向量
Embedding 层只能作为模型的第一层
'''
# model = Sequential([
#     Embedding(2000,32,input_length=50),       #  input_dim = 2000, output_dim = 32
#     Dropout(0.2),
#     Flatten(),
#     Dense(256),
#     Activation('relu'),
#     Dropout(0.25),
#     Dense(10),
#     Activation('softmax') # 分十类
# ])
#------------------------------方法二：Model式模型（使用函数式API的Model类模型）--------------------------------------#

# from keras.layers import Input
# from keras.models import Model
# inputs = Input(shape=(50,))
# x = Embedding(2000,32)(inputs)
# x = Dropout(0.2)(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.25)(x)
# outputs = Dense(10, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=outputs)
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------方法三：Model类继承（class）------------------------------------------------#
import keras
from keras.layers import Input
from numpy import shape
inputs = Input(shape(50))
class mymodel(keras.Model):
    def __init__(self):
        super(mymodel,self).__init__()
        self.Embedding = keras.layers.Embedding(2000,32,input_length=50)
        self.dropout = keras.layers.Dropout(0.2)
        self.Flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256,activation='relu')
        self.dropout = keras.layers.Dropout(0.25)
        self.dense2 = keras.layers.Dense(10, activation='softmax')


    def call(self,inputs):
        x = self.Embedding(inputs)
        x = self.dropout(x)
        x = self.Flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

model= mymodel()

# 报错ValueError: Please provide as model targets either a single array or a list of arrays. You passed: y=[0, 0, 1, 1, 2, 3, 3, 2,
#---------------------------------------------------------------------------------------------------------------------#


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',# 多类对数损失
              metrics = ["accuracy"]
              )

history = model.fit(x_train, y_train,
                    epochs = 5,
                    batch_size = 256,
                    verbose = 2,  # 0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                    validation_split = 0.2  # 训练集的20%用作验证集
                    )
print(model.summary())  # 打印网络结构
#----------------------------保存模型-----------------------------------#
from keras.utils import plot_model
# 保存模型
model.save('model_MLP_text.h5')
# 模型可视化
plot_model(model,to_file = 'model_MLP_text.png',show_shapes = True)

# 加载模型
from keras.models import load_model
model = load_model('model_MLP_text.h5')
# 载入模型预测：输入第一个x,得到第一个预测y
print(x_train[0]) # 输出真实的x
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))  # 输出第一个预测y
print(y_train[0])   # 输出真实的y
# ----------------画图显示运行结果---------------------------------#
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




