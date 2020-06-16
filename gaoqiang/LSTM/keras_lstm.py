# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.16
# 开发框架：keras
# 代码功能：LSTM
# 温馨提示：
#----------------------------------------------------------------------------------------------------------------------#

from keras.layers import *
from keras.models import *

# LSTM
model = Sequential()
model.add(Embedding(3800,32,input_length=380))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


# 双向LSTM(Bi-LSTM)
model = Sequential()
model.add(Embedding(3800,32,input_length=380))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32,return_sequences=True),merge_mode='concat'))
'''
1.return_sequences：布尔值（默认为False），True:返回输出序列中的完整序列(batch_size, timesteps, output_size),
                                         False:返回输出序列最后一个输出(batch_size, output_size)。
只有一层LSTM，一般“return_sequences =False,有多层LSTM，第一层必须加上“return_sequences =True,最后一层一般为
“return_sequences =False”

2.merge_mode='concat' 前向和后向神经网络的输出组合方式。One of {'sum'， 'mul'， 'concat'， 'ave'， None}。
如果为None，则不会合并输出，它们将作为列表返回。默认值是“concat”。

'''
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))