# ----------------开发者信息--------------------------------#
# 开发者：徐珂
# 开发日期：2020年6月4日
# 开发框架：pytorch
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 Keras, matplotlib, numpy, sklearn 的包
# 2、读取数据与图像预处理
# 3、搭建传统CNN模型
# 4、训练模型
# 5、保存模型与模型可视化
#--------------------------------------------------------------#

#  -------------------------- 导入需要包 -------------------------------
from keras import Model
from keras.layers import Input
import matplotlib.pyplot as plt
#  -------------------------- 2、读取数据与图像预处理 -------------------------------

# 数据集和代码放一起即可  定义load_data函数
def load_data():
    paths = [
        'D:/keras/图像分类/train-labels-idx1-ubyte.gz', 'D:/keras/图像分类/train-images-idx3-ubyte.gz', #训练集标签及图像路径
        'D:/keras/图像分类/t10k-labels-idx1-ubyte.gz', 'D:/keras/图像分类/t10k-images-idx3-ubyte.gz',   #测试集标签及图像路径
    ]
#  训练集
    with gzip.open(paths[0], 'rb') as lbpath:                                 # 打开文件 解压训练集标签
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:                                # 解压训练集图像
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
#  测试集
    with gzip.open(paths[2], 'rb') as lbpath:                                  # 打开文件 解压测试集标签
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:                                 # 解压测试集图像
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)                                 # 返回函数

(x_train, y_train), (x_test, y_test) = load_data()                              # 调用load_data函数获取训练集以及测试集
batch_size = 32                                                                 # 批次大小
num_classes = 10                                                                # 需要预测的类的数量
epochs = 5
data_augmentation = True                                                        # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')                        # 保存路径
model_name = 'keras_fashion_trained_model.h5'                                   # 模型名字

# Convert class vectors to binary class matrices. 类别独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)                      # 把标签变成向量形式
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255   # 归一化

#-------------------------------------CNN模型搭建---------------------------------#
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size =3,stride=1,padding=1) # 二维卷积
        self.relu1 = torch.nn.ReLU()                # 激活
        self.Conv2 = torch.nn.Conv2d(32,32,3,1,0)
        self.relu2 = torch.nn.ReLU()
        self.MaxPool1 = torch.nn.MaxPool2d(2)
        self.Dropout1 = torch.nn.Dropout(0.25)
        self.Conv3 = torch.nn.Conv2d(32,64,3,1,1)
        self.relu3 = torch.nn.ReLU()
        self.Conv4= torch.nn.Conv2d(64,64,3,1,1)
        self.relu4 = torch.nn.ReLU()
        self.MaxPool2 = torch.nn.MaxPool2d(2)
        self.Dropout2 = torch.nn.Dropout(0.25)
        self.Flatten = torch.nn.Flatten()           # 拉平
        self.linear1 = torch.nn.Linear(2304,512)
        self.relu2 = torch.nn.ReLU()
        self.Dropout3 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(512, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.Conv1(x)
        x = self.relu1(x)
        x = self.Conv2(x)
        x = self.relu2(x)
        x = self.MaxPool1(x)
        x = self.Dropout1(x)
        x = self.Conv3(x)
        x = self.relu3(x)
        x = self.Conv4(x)
        x = self.relu4(x)
        x = self.MaxPool2(x)
        x = self.Dropout2(x)
        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.Dropout3(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

model = Model()


#  ------------------------- 训练   ---------------------------#
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    # 学习率
loss_fn = torch.nn.MSELoss()
Epoch =5

## 开始训练 ##
for i in range(Epoch):
    x = model(x_train)          # 前向传播
    loss = loss_fn(x, y_train)  # 计算损失

    optimizer.zero_grad()       # 梯度清零
    outputs = net(inputs)       # 数据过网络
    loss = criterion(outputs, labels)  # 计算loss
    loss.backward()             # 反向传播
    optimizer.step()            # 更新参数
