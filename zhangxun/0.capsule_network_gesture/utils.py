import  torch
from    matplotlib import pyplot as plt
# Matplotlib是Python最著名的2D绘图库，该库仿造Matlab提供了一整套相似的绘图函数，
# 用于绘图和绘表，强大的数据可视化工具和做图库，适合交互式绘图，图形美观。
import  random

device = torch.device('cuda')

def plot_curve(data): #绘制下降曲线
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

"""
参数cmap用于设置热图的Colormap。
Colormap是MATLAB里面用来设定和获取当前色图的函数，可以设置如下色图：
    hot 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。
    cool 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。
    gray 返回线性灰度色图。
    bone 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。
    white 全白的单色色图。 
    spring 包含品红和黄的阴影颜色。 
    summer 包含绿和黄的阴影颜色。
    autumn 从红色平滑变化到橙色，然后到黄色。 
    winter 包含蓝和绿的阴影色。
"""

def plot_image(img, label, name, max): #画图片  img, label对应x, y，分别为4维和1维

    fig = plt.figure()
    for k in range(6):
        i = random.randint(0, max)
        plt.subplot(2, 3, k+1) #参数分别为：行数，列数，位置
        plt.tight_layout()
        #img[i+k][0] *0.3081 + 0.1307
        plt.imshow(img[i][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制，对图像进行处理，并显示其格式
        plt.title("{}: {}".format(name, label[i].item())) #用item得到张量的元素值，否则得到的是张量本身
        plt.xticks([])
        plt.yticks([])
    plt.show() #将plt.imshow()处理后的图像显示出来


def one_hot(label, depth=10): #label转onehot （独热码:有多少个状态就有多少位置，每个位置是出现的概率，第一个位置一般表示0
    # 故1., 0., 0., ..., 0., 0., 0.表示是0的概率为1）
    out = torch.zeros(label.size(0), depth).to(device)
    idx = torch.cuda.LongTensor(label).view(-1, 1).to(device)
    out.scatter_(dim=1, index=idx, value=1).to(device)
    return out

def label(one_hot): #onehot转label
    out = torch.topk(one_hot, 1)[1].squeeze(1)
    #topk:将高维数组沿某一维度(该维度共N项),选出最大(最小)的K项并排序。返回排序结果和index信息
    return out