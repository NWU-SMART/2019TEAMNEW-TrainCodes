# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/27
# 文件名称：Keras函数式API.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、Sequential模型和其对应的函数式API
# 3、多输入模型
# 4、多输出模型
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.models import Sequential, Model
from keras import layers
from keras import Input
import numpy as np
from keras.utils.np_utils import to_categorical
#  -------------------------- 1、导入需要的包 -------------------------------
#  -------------------------- 2、Sequential模型和其对应的函数式API -------------------------------
# 学过的Sequential模型
'''
seq_model=Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))

# 对应的函数式API实现
input_tensor=Input(shape=(64,))
x=layers.Dense(32,activation='relu')(input_tensor)
x=layers.Dense(32,activation='relu')(x)
output_tensor=layers.Dense(10,activation='softmax')(x)

# Model将输入张量和输出张量转换为一个模型
model=Model(input_tensor,output_tensor)
# 查看模型
model.summary()

# 如果利用不相关输入和输出来构建一个模型，那么会得到ValueError
#unrelated_input=Input(shape=(32,))
#bad_model=model=Model(unrelated_input,output_tensor)
#  报错：keras无法从给定的输出张量达到input——1
#  ValueError: Graph disconnected: cannot obtain value for tensor Tensor("input_1:0", shape=(None, 64), dtype=float32) at layer "input_1".

#对这种Model实例进行编译、训练或评估时，其API与Sequential模型相同
#编译模型
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
#生成用于训练的虚构Numpy数据
import numpy as np
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))
model.fit(x_train,y_train,epochs=10,batch_size=128)
#评估模型
score=model.evaluate(x_train,y_train)
'''
#  -------------------------- 2、 Sequential模型和其对应的函数式API -------------------------------

#  -------------------------- 3、多输入模型 -------------------------------
# 多输入模型示例：用函数式API实现双输入问答模型
# 典型的问答模型有两个输入：一个子软语言描述的问题和一个文本片段（如新闻文章），后者提供用于回答问题的信息。
# 模型要生成一个回答，最简单的情况是该回答只包含一个词，可以通过对某个预定义的词表做softmax得到
# 将文本输入和问题输入分别编码为表示向量，然后连接这些向量，最后在连接好的表示上添加一个softmax分类器
text_vocabulary_size=10000
question_vocabulary_size=10000
answer_vocabulary_size=500
# 文本输入是一个长度可变的整数序列
text_input=Input(shape=(None,),dtype='int32',name='text')
# 将输入嵌入长度为64的向量
embedded_text=layers.Embedding(
    text_vocabulary_size,64)(text_input)
# 利用LSTM将向量编码为单个向量
encoded_text=layers.LSTM(32)(embedded_text)
# 对问题进行相同的处理（使用不同的层实例）
question_input=Input(shape=(None,),dtype='int32',name='question')
embedded_question=layers.Embedding(
    question_vocabulary_size,32)(question_input)
encoded_question=layers.LSTM(16)(embedded_question)
# 将编码后的文本连起来
concatenated=layers.concatenate([encoded_text,encoded_question],axis=1)
# 在上面添加一个softmax分类器
answer=layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)

# 在模型实例化时，指定两个输入和输出
model = Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 将数据输入到多输入模型中
# 生成虚构的Numpy数据
num_samples = 1000
max_length = 100
text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
answers = np.random.randint(answer_vocabulary_size,size=num_samples)
# 回答是one-hot编码的，不是整数
answers = to_categorical(answers,answer_vocabulary_size)

# 使用输入组成的列表来拟合
# model.fit([text,question],answers,epochs=10,batch_size=128)
# 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
 model.fit({'text':text,'question':question},answers,epochs=10,batch_size=128)
#  -------------------------- 3、多输入模型 -------------------------------


#  -------------------------- 4、多输出模型-------------------------------
#多输出示例：一个网络试图同时预测数据的不同性质，比如一个网络，输入某个匿名人士的一系列社交媒体发帖，然后尝试预测那个人的属性，比如年龄、性别、收入水平
# 用函数式API实现一个三输出模型
vocabulary_size=50000
num_income_groups=10
posts_input=Input(shape=(None,),dtype='int32',name='posts')
embedded_posts=layers.Embedding(256,vocabulary_size)(posts_input)
x=layers.Conv1D(128,5,activation='relu')(embedded_posts)
x=layers.MaxPooling1D(5)(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.MaxPooling1D(5)(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.GlobalMaxPool1D()(x)
x=layers.Dense(128,activation='relu')(x)

# 输出层都具有名称
age_prediction=layers.Dense(1,activation='sigmoid',name='gender')(x)
income_prediction=layers.Dense(num_income_groups,
                               activation='softmax',
                               name='income')(x)
gender_prediction=layers.Dense(1, activation='sigmoid',name='gender')(x)

model=Model(posts_input,
            [age_prediction,income_prediction,gender_prediction])

# 多输出模型的编译选项：多重损失
#model.compile(optimizer='rmsprop',
#              loss=['mse','categorical_crossentropy','binary_crossentropy'])
#与上述写法等效，只有输出层具有名称时才能采用这种写法
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'})
# 严重不平衡的损失贡献会导致模型表示针对单个损失值最大的任务优先进行优化，而不考虑其他任务的优化
# 多输出模型的编译选项：损失加权
# model.compile(optimizer='rmsprop',
#              loss=['mse','categorical_crossentropy','binary_crossentropy'],
#              loss_weights=[0.25,1.,10.])
# 与上述写法等效，只有输出层具有名称时才能采用这种写法
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'},
              loss_weights={'age':0.25,
                            'income':1.,
                            'gender':10.})
# 与多输入模型相同，多输出模型的训练输入数据可以是Numpy数组组成的列表或字典
# 将数据输入到多输出模型中
# 假设age_targets,income_targets,gender_targets都是Numpy数组
# model.fit(posts,[age_targets,income_targets,gender_targets],
#          epochs=10,batch_size=64)
# 与上述写法等效，只有输出层具有名称时才能采用这种写法
model.fit(posts,{'age':age_targets,
                 'income':incom_targets,
                 'gender':gender_targets},
          epochs=10,
          batch_size=64)

#  -------------------------- 4、多输出模型-------------------------------
