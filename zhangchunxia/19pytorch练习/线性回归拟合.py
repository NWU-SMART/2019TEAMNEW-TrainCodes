# -----------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月13日
# 内容:用pytorch实现线性回归并实时显示拟合过程
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch的包
# 2、建立数据集
# 3、建立网络
# 4、训练网络
# 5、训练可视化
# ----------------------   代码布局： ------------------------------------
#----------------------------1、导入需要的包------------------------------
import torch
import torch.nn.functional as F#激励函数都在这
from torch.autograd import Variable
import matplotlib.pyplot as plt
#----------------------------1、导入需要的包------------------------------
#----------------------------2、建立数据集--------------------------------
'''
神经网络是如何通过简单的形式将一群数据用一条线条来表示. 或者说, 
是如何在数据当中找到他们的关系, 然后用神经网络模型来建立一个可以代表他们关系的线条
'''
#首先构建假数据集来模拟真实拟合情况，比如一元二次函数，给y加上噪声可以更真实的展现他
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1),就是在a中指定位置N加上一个维数为1的维度
y = x.pow(2) + 0.2*torch.rand(x.size())# noisy y data (tensor), shape=(100, 1)
x,y=torch.autograd.Variable(x),Variable(y)
#----------------------------2、建立数据集--------------------------------
#----------------------------3、建立网络----------------------------------
class NET(torch.nn.Module):#继承torch的Module
    def __init__(self,n_feature,n_hidden,n_output):
        super(NET,self).__init__()#继承__init__功能
        #定义每层用什么形式
        self.hidden=torch.nn.Linear(n_feature,n_hidden)#隐藏层线性输出
        self.predict=torch.nn.Linear(n_hidden,n_output)#输出层线性输出
    def forward(self,x):  #这是Module的forward功能
        x=F.relu(self.hidden(x))#正向传播输入值，神经网咯分析出输出值，激励函数，隐藏层的线性值
        x=self.predict(x)#输出值
        return x
net=NET(n_feature=1,n_hidden=10,n_output=1)
print(net)#输出net网络架构
#----------------------------3、建立网络----------------------------------
#----------------------------4、训练网络----------------------------------
optimizer=torch.optim.SGD(net.parameters(),lr=0.2)#传入net的所有参数，学习率
loss_func=torch.nn.MSELoss()#计算两者误差
plt.ion()#打开交互模式
for i in range(300):
    prediction=net(x)
    loss=loss_func(prediction,y)
    optimizer.zero_grad()#清空上一步的残余更新参数值
    loss.backward()#误差反向传播, 计算参数更新值
    optimizer.step()#将参数更新值施加到 net 的 parameters 上
# ----------------------------4、训练网络----------------------------------
# ----------------------------5、训练可视化--------------------------------
    if i%5==0:#绘制和显示学习率
        plt.cla()#清除当前图形中的当前活动轴，其他轴不受影响
        plt.scatter(x.data.numpy(),y.data.numpy())#散点图
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
# ----------------------------5、训练可视化--------------------------------
'''
//1.torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，
一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)就是将a中所有为1的维度删掉。
不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。
//2.torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，
比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。a.unsqueeze(N) 就是在a中指定位置N加上一个维数为1的维度。
还有一种形式就是b=torch.unsqueeze(a，N) a就是在a中指定位置N加上一个维数为1的维度

'''


