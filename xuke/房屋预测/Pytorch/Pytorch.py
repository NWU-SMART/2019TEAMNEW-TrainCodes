#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.5.27
#software：pycharm
#项目名称：房价预测（pytorch）


# ----------------------   代码布局： ----------------------
# 1、导入包
# 2、载入数据集
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测


# -----------------------------1.导入包-----------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------2.载入数据集----------------------------#
path = 'D:\\keras\\房屋预测\\boston_housing.npz' #数据集地址#
f = np.load(path)      #numpy.load（）读取数据#
# 404个训练，102个测试
# 训练集
x_train=f['x'][:404]  # 404个训练集（0-403）#
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 102个测试集（404-505）#
y_valid=f['y'][404:]
f.close()   # 关闭文件


# 转成DataFrame格式方便数据处理    DataFrame（panda）是二维的表格#
#训练集#
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
#测试集#
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出训练集的x (前5个)
print(y_train_pd.head(5))  # 输出训练集的y (前5个)


#  -------------------------- 3、数据归一化 -------------------------------#
# MinMaxScaler：[ 0，1 ] ；MaxAbsScaler： [ -1，1 ]#
#训练集归一化#
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化#
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#  -------------------------- 4、模型训练   -------------------------------#

#  ------------------------- 4.1 继承add_module类   ---------------------------#

#Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型#
# 先继承，再构建组件，最后组装 #
class Model(torch.nn.Module):
     def __init__(self):
         super(Model,self).__init__()  # 调用MLP父类Module的构造函数来进行必要的初始化 #
         self.input = torch.nn.Sequential()
         self.input.add_module('dense1',torch.nn.Linear(13,10)) #13-10-15，最后输出一个节点#
         self.input.add_module('relu1',torch.nn.ReLU())
         self.input.add_module('dense2',torch.nn.Linear(10,15))
         self.input.add_module('relu2', torch.nn.ReLU())
         self.input.add_module('dense3',torch.nn.Linear(15,1)) #最后一层不需要加激活函数#

     def forward(self, x):  # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出 #
         x = self.conv(x)
         return x

#  ------------------------- 4.2 封装 torch.nn.Sequential()   ---------------------------#
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 调用MLP父类Module的构造函数来进行必要的初始化 #
        self.layer = torch.nn.Sequential( torch.nn.Linear(13, 10),
                                          torch.nn.Dropout(0.2),
                                          torch.nn.Linear(10, 15),
                                          torch.nn.Linear(15, 1)
                                        )

    def forward(self, x):
        x = self.dense(x)
        return x

#  ------------------------- 4.3 OrderedDict子类   ---------------------------#
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layer = nn.Sequential(OrderedDict([('dense1', torch.nn.Linear(13, 10)),  #三个全连接#
                                                ('dropout', torch.nn.Dropout(0.2)), # dropout #
                                                ('dense2', torch.nn.Linear(10, 15)),
                                                ('dense3', torch.nn.Linear(15, 1))])
                                         )
    def forward(self, x):
        x = self.dense(x)
        return x

#  ------------------------- 4.4 类继承   ---------------------------#
class Model(torch.nn.Module):
     def __init__(self):
         super(Model,self).__init__()
         self.layer = torch.nn.Sequential()
         self.dense1 = torch.nn.Linear(13,10)
         self.dense2 = torch.nn.Linear(10,15)
         self.dense3 = torch.nn.Linear(15, 1)
     def forward(self,x):
         x = self.dense1(x)
         x = self.relu(x)
         x = self.dense2(x)
         x = self.relu(x)
         result = self.dense3(x)
         return result


#  ------------------------- 测试   ---------------------------#
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
Epoch = 200

## 开始训练 ##
for i in range(Epoch):
    x = model(x_train)          # 前向传播
    loss = loss_fn(x, y_train)  # 计算损失

    optimizer.zero_grad()       # 梯度清零
    outputs = net(inputs)       # 数据过网络
    loss = criterion(outputs, labels)  # 计算loss
    loss.backward()             # 反向传播
    optimizer.step()            # 更新参数
print('epoch %3d , loss %3d' % (epoch, batch_loss))