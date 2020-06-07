# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.1
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 代码布局 --------------------*/
# 1.导入需要的包
# 2.读取图片数据
# 3.图片预处理
# 4.构建模型自编码器
# 5.训练
# 6.训练结果可视化
# 7.查看效果
# /------------------ 代码布局 --------------------*/

# /------------------ 导入需要的包--------------------*/
import numpy as np
import matplotlib.pyplot as plt

# /------------------ 导入需要的包--------------------*/

# /------------------ 读取数据--------------------*/
# 数据路径
path = 'mnist.npz'
data = np.load(path)
# ['x_test', 'x_train', 'y_train', 'y_test']
# print(data.files)
# 读取数据
x_train = data['x_train']#(60000, 28, 28)
x_test = data['x_test']#(10000, 28, 28)
data.close()
# 归一化操作
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
# x_train = np.array(x_train)
# x_test = np.array(x_test)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
import torch
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
# /------------------ 读取数据--------------------*/

# /------------------ 建立模型--------------------*/
from torch.nn import Linear,ReLU,Softmax
import torch
class onesingle(torch.nn.Module):
    def __init__(self):
        super(onesingle,self).__init__()
        self.hidden = Linear(in_features=784,out_features=64)
        self.out = Linear(in_features=64,out_features=784)
        self.relu = ReLU()
        self.softmax = Softmax()
    def forward(self,x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        result = self.softmax(x)
        return result
# /------------------ 建立模型--------------------*/
model = onesingle()
import torch
optimize = torch.optim.Adam(model.parameters(),lr=1e-3)
loss = torch.nn.MSELoss()
epoch = 5
for i in range(5):
    result = model(x_train)
    loss_1 = loss(result,x_train)
    print(i,loss_1.item())
    optimize.zero_grad()
    loss_1.backward()
    optimize.step()

# 查看decoder效果
torch.save(model,'onesingle.h5')
model_new = torch.load('onesingle.h5')
for i in range(5):
    new = model_new(x_test)
    new = new.detach().numpy()
    plt.imshow(new[i].reshape(28, 28))
    plt.gray()  # 显示灰度图像
    plt.show()
    # 打印图片显示decoder效果
