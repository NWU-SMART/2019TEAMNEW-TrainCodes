#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月29日
# 开发框架：Pytorch
# 开发内容：使用OSNet做ReID
#----------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------#
import torchreid
#  -------------------------- 1、导入需要包 -------------------------------#

#  -------------------------- 2、Load image data manager -------------------------------#
datamanager = torchreid.data.ImageDataManager(
        root='reid-data',         # root path to datasets
        sources='market1501',     # source dataset(s)
        targets='market1501',     # target dataset(s)（If not given,it equals to ``sources``）
        height=256,               # target image height. Default is 256.
        width=128,                # target image width. Default is 128.
        batch_size_train=32,      # number of images in a training batch. Default is 32.
        batch_size_test=100,      # number of images in a test batch. Default is 32.
        transforms=['random_flip', 'random_crop', 'random_erase'] # transformations applied to model training.Default is 'random_flip'.
    )                                             # 随机反转、随机剪裁和随机擦除
'''
RandomErasing(): Randomly erases an image patch
Reference: Zhong et al. Random Erasing Data Augmentation.
'''
#  -------------------------- Load data manager -------------------------------#

#  -------------------------- 3、Build model, optimizer and lr_scheduler -------------------------------#
model = torchreid.models.build_model(
        name='osnet_x1_0',                      # model name
        num_classes=datamanager.num_train_pids, # number of training identities
        loss='softmax',                # loss function to optimize the model.
        pretrained=False                         # whether to load ImageNet-pretrained weights
    )

# 使用gpu加速
model = model.cuda()

# 优化器，lr=0.065
optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.065
    )

# 调整学习速率，学习率从0.065开始，在150、225和300个epoch时衰减0.1倍
scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[150, 225, 300]
    )
#  -------------------------- Build model, optimizer and lr_scheduler -------------------------------#

#  -------------------------- 4. Build engine -------------------------------#
# Softmax-loss engine for image-reid
engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,          # image data manager
        model,                # 模型
        optimizer=optimizer,  # 优化器
        scheduler=scheduler,  # 调整学习速率
        label_smooth=True     # use label smoothing regularizer
)
#  --------------------------  Build engine -------------------------------#

#  -------------------------- 5. Run training and test -------------------------------#
engine.run(
        save_dir='log/osnet_x1_0',   # directory to save model
        max_epoch=300,               # maximum epoch
        eval_freq=10,                # evaluation frequency.
        print_freq=10,               # print_frequency.
        test_only=False              # train + test
    )
#  --------------------------  Run training and test -------------------------------#