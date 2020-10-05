# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/25 002518:18
# 文件名称：train
# 开发工具：PyCharm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,EarlyStopping


from nets.SSD import SSD300
from nets.ssd_training import Generator, MultiboxLoss
from utils.anchors import get_anchors
from utils.util import BBoxUtility,ModelCheckpoint
import numpy as np

log_dir = "logs/"
annotation_path = '2007_train.txt'

NUM_CLASSES = 21
input_shape = (300, 300, 3)

priors = get_anchors()  # 得到所有先验框

bbox_util = BBoxUtility(NUM_CLASSES, priors)  # 将种类和先验框传入

val_split = 0.1  # 用0.1进行验证

with open(annotation_path) as f:
    lines = f.readlines()  # 得到文本全部的类容

np.random.seed(10101)  # 使得后面生成的随机数相同
np.random.shuffle(lines)  # 打乱lines中的顺序
np.random.seed(None)
num_val = int(len(lines) * val_split)  # 验证集的个数
num_train = len(lines) - num_val

model = SSD300(input_shape,num_classes=NUM_CLASSES)  # 得到SSD300的模型
path = 'model_data/ssd_weights.h5'

# by_name=False 的时候按照网络的拓扑结构加载权重，by_name=True 的时候就是按照网络层名称进行加载
model.load_weights(filepath=path,by_name=True,skip_mismatch=True)  # 得到权重，也可以不得到，但是训练慢，
model.summary()

# 训练参数设置
# tensorboard是可视化tensorflow模型的训练过程的工具,记录所有训练过程
logging = TensorBoard(log_dir=log_dir)
# ModelCheckpoint 保存训练过程中的最佳模型权,
checkpoint = ModelCheckpoint(log_dir+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss',save_weights_only=True,save_best_only=True,period=1)

# 在训练过程中改变学习了率factor表示patience次val_loss不下降时,lr就会变成lr*factor
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,verbose=1)

# 如果6次val_loss没有改变，就停止训练，
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=6,verbose=1)

# print(len(model.layers))
# i = 1
# for layer in model.layers:
#     print('我是第{}层'.format(i),layer.name)
#     i +=1
freeze_layer = 21  # 应该是冻结VGG16
for i in range(freeze_layer):
    model.layers[i].trainable = False

if True:
    BATCH_SIZE = 8
    Lr = 5e-4
    Init_Epoch = 0   # 起始训练次数
    Freeze_Epoch = 50 # 为冻结训练的次数

    gen = Generator(bbox_util,BATCH_SIZE,lines[:num_train],lines[num_train:],
                    (input_shape[0],input_shape[1]),NUM_CLASSES)
    model.compile(optimizer=Adam(lr=Lr),loss=MultiboxLoss(NUM_CLASSES,neg_pos_ratio=3.0).compute_loss)

    # model.fit(gen.generate(True),
    model.fit(gen.noAugment(True),
              steps_per_epoch=num_train // BATCH_SIZE,  # 参数steps_per_epoch是通过把训练图像的数量除以批次大小得出的。例如，有100张图像且批次大小为50，则steps_per_epoch值为2，因为这儿没有batch_size
              validation_data=gen.generate(False),          # 验证集
              validation_steps=num_val // BATCH_SIZE,  # validation_steps: 仅当steps_per_epoch被指定时有用，在验证集上的step总数。
              epochs=Freeze_Epoch,
              initial_epoch=Init_Epoch,  # initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用
              callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            # verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录



    for i in range(freeze_layer):
        model.layers[i].trainable = True
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        BATCH_SIZE = 8
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 100
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]), NUM_CLASSES)

        model.compile(optimizer=Adam(lr=Lr), loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit(gen.generate(True),
                  steps_per_epoch=num_train // BATCH_SIZE,
                  validation_data=gen.generate(False),
                  validation_steps=num_val // BATCH_SIZE,
                  epochs=Epoch,
                  initial_epoch=Freeze_Epoch,
                  callbacks=[logging, checkpoint, reduce_lr, early_stopping])






