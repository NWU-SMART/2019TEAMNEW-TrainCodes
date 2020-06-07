# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.2
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
#
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
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
import torch
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
# /------------------ 读取数据--------------------*/


# /------------------ 多层自编码器模型的建立--------------------*/
from torch.nn import Linear,ReLU,Sigmoid,Sequential
import torch
class encoder(torch.nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.input = Sequential(
            Linear(in_features=784,out_features=128),
            ReLU()
        )
        self.encoder_m = Sequential(
            Linear(in_features=128,out_features=64),
            ReLU()
        )
        self.hidden = Sequential(
            Linear(in_features=64,out_features=128),
            ReLU()
        )
        self.output = Sequential(
            Linear(in_features=128,out_features=784),
            Sigmoid()
        )
    def forward(self,x):
        x = self.input(x)
        x = self.encoder_m(x)
        x = self.hidden(x)
        output = self.output(x)
        return output
model = encoder()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn = torch.nn.MSELoss()
# /------------------ 多层自编码器模型的建立--------------------*/
epoch = 5
for i in range(epoch):
    y_pred = model(x_train)
    loss = loss_fn(y_pred,x_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i,loss.item())
# 查看decoder效果
torch.save(model,'multilayer.h5')
model_new = torch.load('multilayer.h5')
for i in range(5):
    new = model_new(x_test)
new = new.detach().numpy()
for i in range(5):
    img = plt.subplot(2, 5, i + 1)
    plt.imshow(new[i].reshape(28,28))
    img = plt.subplot(2, 5, i + 6)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()  # 显示灰度图像
plt.show()
    # 打印图片显示decoder效果

