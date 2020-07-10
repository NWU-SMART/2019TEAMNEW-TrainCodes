import  torch 
from    torch import nn, optim, autograd
import  numpy as np
import  torchvision
from    torch.nn import functional as F
from    matplotlib import pyplot as plt
import  random
import  os

# h_dim = 400
batchsz = 1

def plot_image(x, name, path): #绘制图片函数

    fig = plt.figure()  # 新建一个画布
    plt.imshow(x[0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
    #plt.title("{}: {} ".format(name, y[0].item())) #设置标题，用item得到张量的元素值，否则得到的是张量本身
    plt.title(name)
    plt.xticks([]) #x轴坐标设置为空
    plt.yticks([]) #y轴坐标设置为空
    plt.savefig(path) #这个savefig一定要放在show之前
    plt.show() #将plt.imshow()处理后的图像显示出来


# 保存目录

save_dir = './saved_figures' # 图片保存目录
if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录

model_dir = './saved_models'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

G_path = os.path.join(model_dir, 'G.h5')
D_path = os.path.join(model_dir, 'D.h5')

name1 = 'contour'
name2 = 'samples'
path1 = os.path.join(save_dir, name1)
path2 = os.path.join(save_dir, name2)
name3 = 'xr'
name4 = 'xf'
path3 = os.path.join(save_dir, name3)
path4 = os.path.join(save_dir, name4)


# xr

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transforms),
                                           batch_size=batchsz,
                                           shuffle=True)

x, _ = next(iter(train_loader)) # torch.Size([1, 1, 28, 28])
x = x.view(-1, 28, 28)
print(x.shape) # torch.Size([1, 28, 28])
plot_image(x, name3, path3)

# 定义G

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
        )

    def forward(self, z):
        output = self.net(z)
        return output

# 定义D

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2), # 返回一个2维值
            nn.Sigmoid() # 转化成0-1的probability
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

# def data_generator():
#
#     scale = 2.
#     centers = [
#         (1, 0),
#         (-1, 0),
#         (0, 1),
#         (0, -1),
#         (1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (1. / np.sqrt(2), -1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), -1. / np.sqrt(2))
#     ]
#     centers = [(scale * x, scale * y) for x, y in centers]
#     while True:
#         dataset = []
#         for i in range(batchsz):
#             point = np.random.randn(2) * .02
#             center = random.choice(centers)
#             point[0] += center[0]
#             point[1] += center[1]
#             dataset.append(point)
#         dataset = np.array(dataset, dtype='float32')
#         dataset /= 1.414  # stdev
#         yield dataset # yield构成无限循环的生成器

    # for i in range(100000//25):
    #     for x in range(-2, 3):
    #         for y in range(-2, 3):
    #             point = np.random.randn(2).astype(np.float32) * 0.05
    #             point[0] += 2 * x
    #             point[1] += 2 * y
    #             dataset.append(point)
    #
    # dataset = np.array(dataset)
    # print('dataset:', dataset.shape)
    # viz.scatter(dataset, win='dataset', opts=dict(title='dataset', webgl=True))
    #
    # while True:
    #     np.random.shuffle(dataset)
    #
    #     for i in range(len(dataset)//batchsz):
    #         yield dataset[i*batchsz : (i+1)*batchsz]


# def generate_image(D, G, xr, epoch): # 可视化工具
#
#     N_POINTS = 128
#     RANGE = 3
#     plt.clf()
#
#     points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
#     points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
#     points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
#     points = points.reshape((-1, 2))
#     # (16384, 2)
#     # print('p:', points.shape)
#
#     # draw contour 二维平面等高线的绘制
#     with torch.no_grad():
#         points = torch.Tensor(points) # [16384, 2]
#         disc_map = D(points).numpy() # [16384]
#     x = y = np.linspace(-RANGE, RANGE, N_POINTS)
#     cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
#     plt.clabel(cs, inline=1, fontsize=10) # inline=True，表示高度写在等高线上
#     # plt.colorbar()
#     plt.savefig(path1)
#     plt.show()
#
#     # draw samples
#     with torch.no_grad():
#         z = torch.randn(batchsz, 2) # [b, 2]
#         samples = G(z).numpy() # [b, 2]
#     plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
#     plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')
#     plt.savefig(path2)
#     plt.show()


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def gradient_penalty(D, xr, xf):

    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1)
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf) # alpha即t, interpolates即x_hat
    interpolates.requires_grad_() # 需要导数信息，这样backward时就会产生导数

    disc_interpolates = D(interpolates)

    # 手动计算梯度
    # create_graph: 用于二阶求导
    # retain_graph: 保存计算图，还可继续backward
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

criterion = nn.CrossEntropyLoss()
fake_label = [0,1]
real_label = [1,0]

def training(xr, G, D, optim_G, optim_D):

    for epoch in range(500): # GAN核心部分

        # 1. train discriminator for k steps
        for _ in range(5):

            # 1.1 real data

            # x = next(data_iter)
            # xr = torch.from_numpy(x) # 把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

            xr = xr.view(batchsz, -1)
            predr = (D(xr)) # predr: [b, 2]
            #lossr = - (predr.mean()) # max D(xr)
            lossr = criterion(predr.mean(), real_label)

            # 1.2 fake data

            z = torch.randn(batchsz, 10) # z: [b, 10]
            xf = G(z).detach() # xf: [b, 10] # stop gradient on G 保证梯度只会传到xf这里，不会传到z，即不会训练G

            predf = (D(xf)) # predf: [b, 2]
            #lossf = (predf.mean()) # min D(xf)
            lossf = criterion(predf.mean(), fake_label)

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp

            optim_D.zero_grad() # 在这一步清零了train Generator中对D部分计算的冗余梯度！
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()

        # 2. train Generator

        z = torch.randn(batchsz, 10)
        xf = G(z)

        predf = (D(xf)) # predf: [b, 2]
        #loss_G = - (predf.mean()) # max D(xf)
        loss_G = criterion(predf.mean(), real_label)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 10 == 0:

            # generate_image(D, G, xr, epoch)
            print(loss_D.item(), loss_G.item())

    # Save model and weights (保存模型)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    print('Saved trained model at %s ' % model_dir)

def main():

    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator()
    D = Discriminator()

    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9)) # gan的经验参数值
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))


    # data_iter = data_generator()
    # print('batch:', next(data_iter).shape)

    if not os.path.isfile(G_path):
        training(x, G, D, optim_G, optim_D)
    else:
        print('model already exist!')
        # load local model and weights (加载模型)
        G.load_state_dict(torch.load(G_path))
        D.load_state_dict(torch.load(D_path))
        print("Created model and loaded weights from file at %s " % model_dir)

    # test
    z = torch.randn(batchsz, 10)
    xf = G(z)
    xf = xf.view(batchsz, 28, 28).detach() #在计算图中的tensor要画出来需要先detach！
    plot_image(xf, name4, path4)

if __name__ == '__main__':
    main()