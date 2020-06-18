# -*- coding: utf-8 -*-
# @Time: 2020/6/16 10:27
# @Author: wangshengkang
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras import applications
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
import os
import argparse
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import cv2

# -----------------------------------代码布局--------------------------------------------
# 1引入keras，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型
# 5保存模型
# 6画出准确率和损失函数的变化曲线
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
plt.switch_backend('agg')  # 服务器没有gui
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='denoiseAE')
# 调用add_argument()方法添加参数
parser.add_argument('--path', default='mnist.npz', type=str, help='the path to dataset')
parser.add_argument('--batchsize', default='32', type=int, help='batchsize')
parser.add_argument('--gpu', default='6', type=str, help='choose which gpu to use')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 选择gpu

path = opt.path
f = np.load(path)
x_train = f['x_train']
x_test = f['x_test']
y_train = f['y_train']
y_test = f['y_test']
f.close()

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20

x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]


def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]
    return np.round(price, 2)


y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))

# ------------------------------------2数据处理-----------------------------------------
# ------------------------------------3建立模型------------------------------------------

# VGG16模型， include_top=False，不包含最后的3个全连接层
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
print(x_train.shape[1:])

# 建立模型
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('linear'))
# VGG+自己的模型
model = Model(inputs=base_model.input, outputs=model(base_model.output))
# VGG16前15层权值固定住
for layer in model.layers[:15]:
    layer.trainable = False

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])
# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型------------------------------------------

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
    )
    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)
    print(x_train.shape[0] / batch_size)

    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10
                                  )
# ------------------------------------4训练模型------------------------------------------
# ------------------------------------5保存模型------------------------------------------
model.summary()
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s' % model_path)
# ------------------------------------5保存模型------------------------------------------
# ------------------------------------6画曲线------------------------------------------
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
# ------------------------------------6画曲线------------------------------------------
