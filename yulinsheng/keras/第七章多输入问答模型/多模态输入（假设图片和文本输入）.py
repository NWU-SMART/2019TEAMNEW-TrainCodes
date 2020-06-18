# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.18
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/



# 单个模型建立
from keras import Sequential,Model
from keras.layers import Embedding,LSTM,Input,concatenate

embedding_size = 1000
# 参考文本进行嵌入
# 首先需要将文本嵌入到相同的维度
# 文本的长度设置为500
text_input = Input(shape=(None,),name='text')
# 进行文本嵌入输出64
text_embedding = Embedding(embedding_size,64)(text_input)
# 进行LSTM编码
encoder_text = LSTM(32)(text_embedding)

# 定义一个cnn卷积层
from keras.layers import Dense,Conv2D,Activation,MaxPool2D,BatchNormalization,Flatten
input_image = Input(shape=(128,128,3))
cnn_result = Conv2D(16,(3, 3), padding="same",activation='relu')(input_image)
cnn_result = Activation("relu")(cnn_result)
cnn_result = BatchNormalization()(cnn_result)
cnn_result = MaxPool2D(pool_size=(2, 2))(cnn_result)

cnn_result = Conv2D(32,(3, 3), padding="same",activation='relu')(cnn_result)
cnn_result = Activation("relu")(cnn_result)
cnn_result = BatchNormalization()(cnn_result)
cnn_result = MaxPool2D(pool_size=(2, 2))(cnn_result)


cnn_result = Conv2D(64,(3, 3), padding="same",activation='relu')(cnn_result)
cnn_result = Activation("relu")(cnn_result)
cnn_result = BatchNormalization()(cnn_result)
cnn_result = MaxPool2D(pool_size=(2, 2))(cnn_result)

# 由于现在处理后的图片是多维度的没办法和多层感知器处理后的图片
cnn_result = Flatten()(cnn_result)
cnn_result = Dense(units=16,activation='relu')(cnn_result)
feature = concatenate([encoder_text,cnn_result],axis=-1)

result = Dense(units=4,activation='relu')(feature)
result = Dense(units=1,activation='linear')(result)
model = Model(inputs=[text_input,input_image],outputs=result)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)