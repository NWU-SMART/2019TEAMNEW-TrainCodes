# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.4
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"#在第二个GPU上工作
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
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
x_train = torch.tensor(x_train).cuda()
x_test = torch.tensor(x_test).cuda()
# /------------------ 读取数据--------------------*/

# /------------------ 正则自编码器模型的建立--------------------*/
import torch
from torch.nn import Linear,ReLU,Sigmoid,Sequential
class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.input = Sequential(
            Linear(out_features=64,in_features=784),
            ReLU()
        )
        self.out = Sequential(
            Linear(in_features=64,out_features=784),
            Sigmoid()
        )
    def forward(self,x):
        x = self.input(x)
        decoder = self.out(x)
        return decoder
coder = model().cuda()
optimize = torch.optim.Adam(coder.parameters(),lr=1e-4)
loss_fn = torch.nn.MSELoss()

# /------------------ 正则自编码器模型的建立--------------------*/

# /------------------ 模型训练--------------------*/

for i in range(5):
    predict = coder(x_train)
    loss = loss_fn(predict,x_train)
    loss_re = 0
    for param in coder.parameters():
        loss_re += torch.sum(abs(param))
    loss_zong = loss + loss_re*(1e-5)
    print(i,loss_zong.item())
    optimize.zero_grad()
    loss_zong.backward()
    optimize.step()

# 查看decoder效果
torch.save(coder,'编码.h5')
model_new = torch.load('编码.h5').cuda()
import matplotlib.pyplot as plt
# 打印图片显示decoder效果
new = model_new(x_test).cpu()
new = new.detach().numpy()
x_test = x_test.cpu()
n = 5
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(new[i].reshape(28,28))
    img = plt.subplot(2, 5, i + 6)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()#显示灰度图像
    plt.savefig('zhengze.jpg')
plt.show()

