import cv2
import numpy as ny
from keras.layers import concatenate
from keras.utils import to_categorical
from keras.layers import Input
from DataLoad import *
from keras import optimizers, Sequential

#加载数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, XTrainImage, yXTrain  = load_dataset()

#-------------------------------对数据进行归一化--------------------------
#对训练数据进行标准化
X_train = []
for i in range(1):
    files = X_train_orig[:]
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        X_train.append(img)
X_train = np.array(X_train, dtype='float') / 255

#对测试数据进行标准化
X_test = []
for i in range(1):
    files = X_test_orig[:]
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        X_test.append(img)
X_test = np.array(X_test,dtype='float') / 255

#对预训练训练数据集进行标准化
XRay_train = []
for i in range(1):
    files = XTrainImage[:]
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        XRay_train.append(img)
XRay_train = np.array(XRay_train,dtype='float') / 255

"""
#对预训练测试数据集进行标准化
XRay_test = []
for i in range(1):
    files = XTestImage[:]
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        XRay_test.append(img)
XRay_test = np.array(XRay_test,dtype='float') / 255
"""


#对训练标签和测试标签转换为独热编码
Y_train = to_categorical(Y_train_orig)
Y_test = to_categorical(Y_test_orig)
YXTrain = to_categorical(yXTrain)
#YXTest = to_categorical(yXTest)

#-----------------------------数据处理完毕------------------------------
print ("number of Xtraining examples = " + str(XRay_train.shape[0]))  #输出训练集样本的数量
#print ("number of Xtest examples = " + str(XRay_test.shape[0]))       #输出测试集样本的数量
print ("number of training examples = " + str(X_train.shape[0]))  #输出训练集样本的数量
print ("number of test examples = " + str(X_test.shape[0]))       #输出测试集样本的数量
print ("X_train shape: " + str(X_train.shape))   #输出训练数据样本的shape
print ("Y_train shape: " + str(Y_train.shape))   #输出训练样本标签的shape
print ("X_test shape: " + str(X_test.shape))     #输出测试集样本的shape
print ("Y_test shape: " + str(Y_test.shape))     #输出测试样本标签的shape




