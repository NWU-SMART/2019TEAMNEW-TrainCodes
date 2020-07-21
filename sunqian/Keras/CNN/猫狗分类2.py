# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/30
# 文件名称：猫狗分类2.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 使用数据增强来解决猫狗分类实验中的过拟合问题
# 1、导入需要的包
# 2、导入数据
# 3、数据增强
# 4、构建模型
# 5、训练并保存模型
# 6、模型可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import os
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image # 图像预处理工具的模块
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入数据 -------------------------------
# 保存较小数据集的目录
base_dir = 'F:\\cats_and_dogs_small'
# 划分后的训练目录
train_dir = os.path.join(base_dir, 'train')
# 划分后的验证目录
validation_dir = os.path.join(base_dir, 'validation')
# 划分后的测试目录
test_dir = os.path.join(base_dir, 'test')
# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
#  -------------------------- 2、导入数据-------------------------------

#  -------------------------- 3、数据增强-------------------------------

# 数据增强时从现有的训练样本中生成更多的训练模型，其方法是利用多种能够生成可信图像的随机变换来增加（augment）
# 利用ImageDataGenerator来设置数据增强
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# 显示几个随机增强后的训练图像
fnames = [os.path.join(train_cats_dir,fname) for
          fname in os.listdir(train_cats_dir)]
# 选择一张图像进行增强
img_path = fnames[3]
# 读取图像并调整大小
img = image.load_img(img_path,target_size=(150,150))
# 将其转换为形状(150,150,3)的Numpy数组
x = image.img_to_array(img)
# 将其形状改变为(1,150,150,3)
x = x.reshape((1,)+x.shape)
# 生成随机变换狗的图像批量。循环是无限的，所以需要在某个时刻终止循环
i = 0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
#  -------------------------- 3、数据增强-------------------------------
#  -------------------------- 4、构建模型 -------------------------------
# 将猫狗分类的小型卷积神经网络实例化
# 模型中特征图的深度在逐渐增大（从32增大到128），而特征图的尺寸在逐渐减小（从150*150减小到7*7）
# 为了进一步降低过拟合，向模型中添加一个Dropout层，添加到密集连接分类器之前
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
# 添加dropout
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))
# 这是二分类问题，所以模型最后一层使用sigmoid激活的单一单元（大小为1的Dense层）
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

# 编译模型
# 由于模型最后一层是单一单元，采用二元交叉熵作为损失函数
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
#  -------------------------- 4、构建模型 -------------------------------

#  -------------------------- 5、训练并保存模型-------------------------------
# 利用数据增强生成器训练卷积神经网络
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
# 不能增强验证数据
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

# 利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
# 保存模型
model.save('cats_and_dogs_small_2.h5')

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
plt.legend(['Train', 'Valid'], loc='upper left') # 左上方显示图例
plt.savefig('Valid_acc.png')
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
#  -------------------------- 6、模型可视化-------------------------------



