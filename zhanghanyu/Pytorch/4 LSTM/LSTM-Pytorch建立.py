# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月24日
# 内容：LSTM-pytorch
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入包
# 2、模型建立
# 3、模型训练
# ----------------------   代码布局： ----------------------
# ------------------------1. 导入包-------------------------
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
# ------------------------1. 导入包-------------------------
# Parameters 是 Variable 的子类。Paramenters和Modules一起使用的时候会有一些特殊的属性，
# 即：当Paramenters赋值给Module的属性的时候，他会自动的被加到 Module的参数列表中(会出现
# 在 parameters() 迭代器中)。将Varibale赋值给Module属性则不会有这样的影响。 这样做的原
# 因是：我们有时候会需要缓存一些临时的状态(state), 比如：模型中RNN的最后一个隐状态。如果
# 没有Parameter这个类的话，那么这些临时变量也会注册成为模型变量。
# ------------------------2. 模型定义-------------------------
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate  输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))   # parameter是在定义函数function的时候，传到函数的参数：没有固定的值；
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # forget gate 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # output gate 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell  cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))
        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs: Tensor, state: tuple[Tensor]) \
            -> tuple[Tensor, tuple[Tensor, Tensor]]:   # ->用来提示该函数返回值的数据类型
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        #         seq_size, batch_size, _ = inputs.size()

        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        seq_size = 1
        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +
                              self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)   # t（）矩阵转置
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)

def reset_weigths(model):
    """reset weights
    """
    for weight in model.parameters():
        init.constant_(weight, 0.5)
# ------------------------2. 模型定义-------------------------

# ---------------------------3. 测试--------------------------
### test
inputs = torch.ones(1, 1, 10)
h0 = torch.ones(1, 1, 20)
c0 = torch.ones(1, 1, 20)
print(h0.shape, h0)
print(c0.shape, c0)
print(inputs.shape, inputs)

# test naivelstm with input_size=10, hidden_size=20
naive_lstm = LSTM(10, 20)
reset_weigths(naive_lstm)
output1, (hn1, cn1) = naive_lstm(inputs, (h0, c0))

print(hn1.shape, cn1.shape, output1.shape)
print(hn1)
print(cn1)
print(output1)

# --------------------------3. 测试---------------------------
