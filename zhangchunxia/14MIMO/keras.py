# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月30日
# 内容:MIMO
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、函数功能区
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、函数功能区 -----------------------------------
'''
keras中文官方文档的例子：
主要输入(main_input): 新闻标题本身，即一系列词语.
辅助输入(aux_input): 接受额外的数据，例如新闻标题的发布时间等.
该模型将通过 两个损失函数 进行监督学习.
较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法.
'''
# 定义网络模型
# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间
# 注意我们可以通过传递一个 `name` 参数来命名任何层
main_input = Input(shape=(100,), dtype='int32', name='main_input')
# Embedding 层将输入序列编码为一个稠密向量的序列，每个向量维度为 512
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
# LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)
# 在这里添加辅助损失，使得即使在模型主损失很高的情况下，LSTM层和Embedding层都能被平稳地训练
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)#第一个输出
# 此时，我们将辅助输入数据与LSTM层的输出连接起来,输入到模型中
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
#再添加剩余的层
# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)#另一个输出
# 定义这个具有两个输入和输出的模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
# 编译模型时候分配损失函数权重：编译模型的时候，给辅助损失 分配一个0.2的权重
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
#loss_weight:用来计算总的loss 的权重。默认为1，多个输出时，可以设置不同输出loss的权重来决定训练过程
# 训练模型：我们可以通过传递输入数组和目标数组的列表来训练模型
model.summary()
plot_model(model,to_file='D:/northwest/小组视频model.png')
#  ---------------------- 2、函数功能区 -----------------------------------
