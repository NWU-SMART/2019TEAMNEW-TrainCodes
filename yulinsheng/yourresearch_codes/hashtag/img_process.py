
'''
读取图片数据,调用VGG16处理

'''
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array,load_img
import h5py
import time
import os

# 首先读取已经已经训练好的VGG16网络
model = VGG16(weights='imagenet',include_top=False)
# 图片数据地址
datapath = os.getcwd()+'/data_shiyan/'
# 归一化处理后的平均像素
mean_pixel = [103.939, 116.779, 123.68]
# 获取图片的名称
img_list = os.listdir(datapath)
# print(img_list)
count = 0
# 记录时间
start = time.time()
for item in img_list[:]:
    name = datapath + item
    print(name)
    try:
        # image.load_img()只是加载了一个文件，没有形成numpy数组
        img = load_img(name, target_size=(224, 224))
        # 将读取的数据转换为数组类型
        img_data = img_to_array(img)
        #扩展数组维度行扩展
        '''
        如（2,3）的矩阵，axis=0是，（1,2,3）
        axis (2,1,3)
        '''
        img_data = np.expand_dims(img_data,axis=0)
        # 对数据进行预处理
        img_data = preprocess_input(img_data)
    #     利用预先下载好的模型进行训练
        img_process = model.predict(img_data)

    except:
        print('发生错误')
        continue
# (1,224,224,3)
print(img_data.shape)
# (1,7,7,512)
print(img_process.shape)