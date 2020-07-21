# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/29
# 文件名称：猫狗分类1.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、将图像复制到训练、验证、和测试的目录
# 3、构建模型
# 4、数据预处理
# 5、训练模型并保存
# 6、模型可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import os
import shutil
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、将图像复制到训练、验证、和测试的目录 -------------------------------
# 原始数据集解压目录的路径
original_dataset_dir = 'F:\\kaggle_original_data'
#保存较小数据集的目录
base_dir = 'F:\\cats_and_dogs_small'
os.mkdir(base_dir)
# 划分后的训练目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
# 划分后的验证目录
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# 划分后的测试目录
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
# 将前1000张猫的图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来500张猫的图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
# 将接下来的500张猫的图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


# 将前1000张狗的图像复制到train_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# 将接下来500张狗的图像复制到validation_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
# 将接下来的500张狗的图像复制到test_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst=os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
# 检查每个分组（训练/验证/测试）中分别包含多少张图像
# print('猫的训练图片总数:',len(os.listdir(train_cats_dir)))
# print('狗的训练图片总数：',len(os.listdir(train_dogs_dir)))
# print('猫的验证图片总数：',len(os.listdir(validation_cats_dir)))
# print('狗的验证图片总数：',len(os.listdir(validation_dogs_dir)))
# print('猫的测试图片总数：',len(os.listdir(test_cats_dir)))
# print('狗的测试图片总数：', len(os.listdir(test_dogs_dir)))
#  -------------------------- 2、将图像复制到训练、验证、和测试的目录-------------------------------

#  -------------------------- 3、构建模型 -------------------------------
# 将猫狗分类的小型卷积神经网络实例化
# 模型中特征图的深度在逐渐增大（从32增大到128），而特征图的尺寸在逐渐减小（从150*150减小到7*7）
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
# 这是二分类问题，所以模型最后一层使用sigmoid激活的单一单元（大小为1的Dense层）
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

# 编译模型
# 使用RMSprop优化器。由于模型最后一层是单一单元，采用二元交叉熵作为损失函数
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
#  -------------------------- 3、构建模型 -------------------------------

#  -------------------------- 4、数据预处理-------------------------------
# 数据预处理的步骤：读取图像文件；将jpg文件解码为RGB像素网格；将这些像素网格转换为浮点数张量；将像素值（0-255）缩放到[0,1]区间
# 使用ImageDataGenerator从目录中读取图像
# 将所有图像乘以1/255缩放
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),# 将所有图像的大小调整为150*150
    batch_size=20,
    class_mode='binary') # 由于使用了binary_crossentropy损失，所以用二进制标签
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')
# 查看生成器的输出：data batch shape:(20,150,150,3) labels batch shape:(20,)
# 生成了150*150的RGB图像（形状为(20,150,150,3)） 与二进制标签（形状为(20,)）组成的批量
# for data_batch, labels_batch in train_generator:
#    print('data batch shape:',data_batch.shape)
#    print('labels batch shape:',labels_batch.shape)
#    break

#  -------------------------- 4、数据预处理-------------------------------

#  -------------------------- 5、训练并保存模型-------------------------------
# 利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
# 保存模型
model.save('cats_and_dogs_small_1.h5')

#  -------------------------- 5、训练并保存模型 -------------------------------

#  -------------------------- 6、模型可视化-------------------------------
# 绘制训练过程中的损失曲线和精度曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 从图像中可以看出过拟合
# 训练精度随着时间线性增加，直到接近100%，而验证精度则停留在70%-72%。
# 验证损失仅在5轮后就达到最小值，然后保持不变，而训练损失则一直现行下降，直到接近于0
# 为了降低过拟合，将使用计算机视觉领域的新方法：数据增强来解决过拟合问题。
#  -------------------------- 6、模型可视化-------------------------------



