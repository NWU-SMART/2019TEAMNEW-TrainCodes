# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.16
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 网络层设置 --------------------*/
#输入问题和一段文字描述
# 文本描述是参考文本
from keras.models import Model
from keras.layers import Dropout,Dense,Input,Embedding,LSTM,concatenate
# 假设嵌入大小为10000
embedding_size = 1000


# 参考文本进行嵌入
# 首先需要将文本嵌入到相同的维度
# 文本的长度设置为500
text_input = Input(shape=(None,),name='text')
# 进行文本嵌入输出64
text_embedding = Embedding(embedding_size,64)(text_input)
# 进行LSTM编码
encoder_text = LSTM(32)(text_embedding)
# 问题文本嵌入
question_input = Input(shape=(None,),name='question')
question_embedding = Embedding(embedding_size,32)(question_input)
encoder_question = LSTM(16)(question_embedding)
# 将两种特征连接到一块,axis=-1为去掉所有[]拼接到一块
features = concatenate([encoder_question,encoder_text],axis=-1)
answer_size = 500
# 定义输出利用softmax函数
answer = Dense(500,activation='softmax')(features)
model = Model([text_input,question_input],answer)
# /------------------ 网络层设置 --------------------*/

# /------------------模型参数选择 --------------------*/
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
# /------------------ 模型参数选择 --------------------*/

# /------------------ 输出模型结构 --------------------*/
# 打印模型结构
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model1.png',show_shapes=True)
# /------------------ 输出模型结构 --------------------*/


# /------------------ 伪造数据训练--------------------*/
num_samples = 1000
max_length = 100
import numpy as np
from keras.utils import to_categorical
# 伪造数据
text = np.random.randint(1, embedding_size, size=(num_samples, max_length))
question = np.random.randint(1, embedding_size, size=(num_samples, max_length))
# 随机生成结果并将结果进行one-hot编码
answers = np.random.randint(answer_size, size=num_samples)
answers = to_categorical(answers, answer_size) # one-hot化
#训练模型
result = model.fit([text,question],answers,
          epochs=10,batch_size=128)
import matplotlib.pyplot as plt
plt.plot(result.history['loss'])
plt.plot(result.history['acc'])
plt.show()

# /------------------ 伪造数据训练--------------------*/