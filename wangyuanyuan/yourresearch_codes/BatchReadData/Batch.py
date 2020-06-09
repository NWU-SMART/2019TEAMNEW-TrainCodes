import os
import cv2
import pandas as pd
import numpy as np
import numpy
from keras import Input, Model, optimizers
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical, multi_gpu_model



class DataGenerator(Sequence):

    def __init__(self, files, batch_size=64, shuffle=True):
        self.batch_size = batch_size
        self.files = files
        self.indexes = numpy.arange(len(self.files))
        self.shuffle = shuffle

    def __len__(self):
        """return: steps num of one epoch. """
        return len(self.files) // self.batch_size

    def __getitem__(self, index):
        print(index)
        batch_inds = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        # get batch data file name.
        batch_files = [self.files[k] for k in batch_inds]

        # read batch data
        batch_images, batch_labels = self._read_data(batch_files)

        return batch_images, batch_labels

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)

    def _read_data(self, batch_files):
        global num_classes
        images = []
        array1 = []
        data = pd.read_csv('Test.csv', encoding='UTF-8')
        label = list(data['Finding Labels'].unique())

        def label_dataset(row):
            num_label = label.index(row)
            return num_label
        data['label'] = data['Finding Labels'].apply(label_dataset)
        yt = data.values


        for file in batch_files:
            image = cv2.imread('D:/SoftWare/Pycharm/Xray/images_012/' + file + '.png')
            image = cv2.resize(image, (512, 512))
            images.append(image)
            name = file + '.png'
            for y in yt:
                if y[0]==name:
                    array1.append(y[-1])
        images = np.array(images, dtype='float') / 255
        array1 = to_categorical(array1)
        return images, array1


def get_generator(batch_size=64, preprocess=True, shuffle=True):

    images_files = os.listdir('D:\SoftWare\Pycharm\Xray\images_012')
    allfiles = []
    for file in images_files:
        allfiles.append(file.split('.')[0])  # get image name
    N = len(allfiles)
    np.random.shuffle(allfiles)
    train_files = allfiles[:]
    #test_files = allfiles[int(N*train_ratio):]
    generator = DataGenerator(train_files, batch_size, shuffle)
    return generator
generator = get_generator(batch_size=64, preprocess=True, shuffle=True)


inputs_all = Input(shape=(512, 512, 3))
x = Conv2D(16, (2, 2), padding='same', name='conv1')(inputs_all)
x = Flatten()(x)
out = Dense(266, activation='softmax', name='fc5')(x)
model = Model(inputs = inputs_all, outputs = out)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  #模型训练的BP模式设置
model.summary()
model.fit_generator(generator, epochs=2)