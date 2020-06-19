#---------------------------------------------------------开发者信息-----------------------------------------------------
#开发者：王园园
#开发时间：2020.6.18
#开发软件：pycharm
#开发项目：多输入多输出(多任务模型)（keras）

#------------------------------------------------------------导包--------------------------------------------------------
import keras
from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense

#-------------------------------------------------------------定义网络模型-----------------------------------------------
#定义网络模型
#标题输入：接收一个含有100个整数的序列，每个整数在1到10000之间
#我们可以通过传递一个'name'参数来命名任何层
main_input = Input(shape=(100,), dtype='int32', name='main_input')
#Embedding层将输入序列编码为一个稠密向量的序列，每个向量维度为512
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#LSTM层把向量序列转换成单个向量，它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)

#在这里我们添加辅助损失，使得即使在模型主损失很高的情况下，LSTM层和Embedding层都能被平稳地训练
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
#此时，我们将辅助输入数据与LSTM层的输出连接起来，输入到模型中
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_output])

#在添加剩余的层
#堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

#定义这个具有两个输入和输出的模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
#编译模型的时候分配损失函数权重：编译模型的时候，给辅助损失分配一个0.2的权重
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
#训练模型：我们可以通过传递输入数组和目标数组的列表来训练模型
model.fit([headline_data, additional_data], [labels, labels], epochs=50, batch_size=32)
#另外一种利用字典的编译、训练方式
#由于输入和输出均被命名了（在定义时传递了一个name参数），我们也可以通过以下方式编译模型
model.compile(optimizer='rmsprop',
              loss={'main_output':'binary_crossentropy', 'aux_output':'binary_crossentropy'},
              loss_weights={'main_output':1., 'aux_output':0.2})
#然后使用一下方式训练
model.fit({'main_input':headline_data, 'aux_input':additional_data},
          {'main_output':labels, 'aux_output':labels},
          epochs=50, batch_size=32)





