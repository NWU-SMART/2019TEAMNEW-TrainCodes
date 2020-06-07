import glob
import os
import numpy as np
import tensorflow as tf
import h5py
import math
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def load_dataset():

    if True:
        #数据增强
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,      #随机修剪
            zoom_range=0.01,    #随机变焦
            brightness_range=[0.1, 1],  #随机亮度调整
            horizontal_flip=True,  # 进行随机水平翻转
            vertical_flip=True,  # 进行随机竖直翻转
            fill_mode='nearest')

        # 非新冠肺炎训练数据集数据增强
        path = "TrainData/COVIDNON-traindata/";
        dirs = os.listdir(path)
        nonTraindata = []
        for file in dirs:
            img = load_img(path + file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，
            nonTraindata.append(x)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='TrainData/COVIDNON-traindata/',
                                      save_prefix='0', save_format='png'):
                i += 1
                if i > 10:  # 数据扩充倍数，此处为数据扩充20倍
                    break  # 否则生成器会退出循环

        # 新冠肺炎训练数据增强
        path = "TrainData/COVID-traindata/";
        dirs = os.listdir(path)
        Traindata = []
        for file in dirs:
            img = load_img(path + file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，
            Traindata.append(x)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='TrainData/COVID-traindata/',
                                      save_prefix='1', save_format='png'):
                i += 1
                if i > 10:  # 数据扩充倍数，此处为数据扩充20倍
                    break  # 否则生成器会退出循环

        # 提取非新冠肺炎训练数据集标签
        path = "TrainData/COVIDNON-traindata/";
        dirs = os.listdir(path)
        nonCOVIDtraindata = []
        for file in dirs:  ## 从训练集中获取图片的文件名
            label = int(file.split("_")[0])
            nonCOVIDtraindata.append(label)
        print('非新冠训练数据集大小：')
        # print(nonCOVIDValdata)
        print(nonCOVIDtraindata.__len__())

        # 提取新冠训练数据集标签
        path = "TrainData/COVID-traindata/";
        dirs = os.listdir(path)
        COVIDtraindata = []
        for file in dirs:  ## 从训练集中获取图片的文件名
            label = int(file.split("_")[0])
            COVIDtraindata.append(label)
        print('新冠训练数据集大小：')
        print(COVIDtraindata.__len__())

        # 训练数据集及其标签
        Train_NONimages = sorted(glob.glob('TrainData/COVIDNON-traindata/*.png'))  # 加载训练数据
        Train_COVIDimages = sorted(glob.glob('TrainData/COVID-traindata/*.png'))
        Train_images = Train_NONimages + Train_COVIDimages
        y = nonCOVIDtraindata + COVIDtraindata
        #加载测试数据及其标签
        Val_images = sorted(glob.glob('TestData/valdata/*.png'))
        path = "valData.csv"  # 加载训练数据的标签
        dft = pd.read_csv(path)
        yt = dft.values
        yt = yt[:, 1]

        #预训练数据集
        XImage1 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_001/*.png'))
        XImage2 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_002/*.png'))
        XImage3 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_003/*.png'))
        XImage4 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_004/*.png'))
        XImage5 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_005/*.png'))
        XImage6 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_006/*.png'))
        #XImage7 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_007/*.png'))
        #XImage8 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_008/*.png'))
        #XImage9 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_009/*.png'))
        #XImage10 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_010/*.png'))
        #XImage11 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_011/*.png'))
        #XImage12 = sorted(glob.glob('/home/wangyuanyuan/projects/x14data/images_012/*.png'))
        XTrainImage = XImage1+XImage2+XImage3+XImage4+XImage5+XImage6

        #测试数据集

        #XTestImage = XImage12

        #预训练训练数据集的标签
        data = pd.read_csv('Data_Entry.csv', encoding='UTF-8')
        label = list(data['Finding Labels'].unique())
        def label_dataset(row):
            num_label = label.index(row)
            return num_label
        data['label'] = data['Finding Labels'].apply(label_dataset)
        value = data.values
        yXTrain = value[:, -1]
        """
        #预训练测试集标签
        data1 = pd.read_csv('Test.csv', encoding='UTF-8')
        label1 = list(data1['Finding Labels'].unique())
        def label_dataset(row):
            num_label = label1.index(row)
            return num_label
        data1['label1'] = data1['Finding Labels'].apply(label_dataset)
        value1 = data1.values
        yXTest = value1[:, 2]    #测试数据集标签
        #print(value[0:104998, 12])
        """



    # 将训练数据集与其标签相对应
        np.random.seed(10)
        np.random.shuffle(Train_images)
        np.random.seed(10)
        np.random.shuffle(y)
    #将测试集与其标签顺序打乱后也对应
        np.random.seed(10)
        np.random.shuffle(Val_images)
        np.random.seed(10)
        np.random.shuffle(yt)
    #将预训练训练数据集与其标签在打乱后对应
        np.random.seed(10)
        np.random.shuffle(XTrainImage)
        np.random.seed(10)
        np.random.shuffle(yXTrain)
        """
        #将预训练测试数据集与其标签在打乱后相对应
        np.random.seed(10)
        np.random.shuffle(XTestImage)
        np.random.seed(10)
        np.random.shuffle(yXTest)
        """

    train_set_x_orig = Train_images
    train_set_y_orig = y

    #test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = Val_images
    test_set_y_orig = yt

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, XTrainImage, yXTrain

