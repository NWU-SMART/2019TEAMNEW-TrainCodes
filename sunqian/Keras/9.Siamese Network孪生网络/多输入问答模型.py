# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/23
# 文件名称：多输入问答模型.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、构建多输入模型
# 3、输入伪造的数据
# 4、训练模型
# 5、模型可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.models import  Model
from keras import Input
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, LSTM, concatenate, Dense
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、构建多输入模型 -------------------------------
# 多输入模型示例：用函数式API实现双输入问答模型
# 典型的问答模型有两个输入：一个自然语言描述的问题和一个文本片段（如新闻文章），后者提供用于回答问题的信息
# 模型要生成一个回答，最简单的情况是该回答只包含一个词，可以通过对某个预定义的词表做softmax得到
# 将文本输入和问题输入分别编码为表示向量，然后连接这些向量，最后在连接好的表示上添加一个softmax分类器
text_vocabulary_size=10000
question_vocabulary_size=10000
answer_vocabulary_size=500

# 文本输入是一个长度可变的整数序列
text_input=Input(shape=(None,),dtype='int32',name='text')

# 将输入嵌入长度为64的向量
embedded_text=Embedding(text_vocabulary_size,64)(text_input)

# 利用LSTM将向量编码为单个向量
encoded_text=LSTM(32)(embedded_text)

# 对问题进行相同的处理（使用不同的层实例）
question_input=Input(shape=(None,),dtype='int32',name='question')

embedded_question=Embedding(question_vocabulary_size,32)(question_input)
encoded_question=LSTM(16)(embedded_question)

# 将编码后的文本连起来
concatenated=concatenate([encoded_text,encoded_question],axis=1)
# 在上面添加一个softmax分类器
answer=Dense(answer_vocabulary_size,activation='softmax')(concatenated)

# 在模型实例化时，指定两个输入和输出
model = Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
# 打印模型结构
model.summary()
#  -------------------------- 2、构建多输入模型 -------------------------------

#  -------------------------- 3、输入伪造的数据-------------------------------
# 将数据输入到多输入模型中
# 生成虚构的Numpy数据
num_samples = 1000
max_length = 100
text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
answers = np.random.randint(answer_vocabulary_size,size=num_samples)
# 回答是one-hot编码的，不是整数
answers = to_categorical(answers,answer_vocabulary_size)
#  -------------------------- 3、输入伪造的数据-------------------------------

#  -------------------------- 4、训练模型-------------------------------
# 使用输入组成的列表来拟合
# model.fit([text,question],answers,epochs=10,batch_size=128)
# 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
result = model.fit({'text':text,'question':question},answers, epochs=10, batch_size=128)
#  -------------------------- 4、训练模型 -------------------------------

#  -------------------------- 5、模型可视化-------------------------------
import matplotlib.pyplot as plt
plt.plot(result.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(result.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
#  -------------------------- 5、模型可视化-------------------------------
