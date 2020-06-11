# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.10
# 开发框架：keras
# 适用场景：如果训练网络时，针对的是大规模数据集，如图像数据集，其不能完全读取加载到内存里，那么就需要利用用到
#           data generator了. data generator 将数据分为 batches，再送入网络进行训练.
#----------------------------------------------------------------------------------------------------------------------#
'''
方法：采用 Keras 的 Sequence Class
    每个 Sequence 必须包含 __getitem__ 和 __len__ 方法的实现.
'''
from keras.utils import Sequence
import numpy as np
from skimage.transform import resize
from skimage.io import imread

class myGenerator(Sequence):
    def __init__(self,image_filenames,labels,batch_size):
        # image_filenames - 图片路径
        # labels - 图片对应的类别标签
        self.image_filenames,self.labels = image_filenames,labels
        self.batch_size = batch_size

    def __len__(self):
        # 计算 generator要生成的 batches 数
        return np.ceil(len(self.image_filenames)/float(self.batch_size))

    def __getitem__(self,index):
        # index - 给定的 batch 数，以构建 batch 数据 [images_batch, GT]
        batch_x = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name),(200,200))
                         for file_name in batch_x]),np.array(batch_y)

# generator 的使用
# my_training_batch_generator = myGenerator(training_filenames,
#                                            GT_training,
#                                            batch_size)
# my_validation_batch_generator = myGenerator(validation_filenames,
#                                              GT_validation,
#                                              batch_size)
#
# model.fit_generator(generator=my_training_batch_generator,
#                     steps_per_epoch=(num_training_samples // batch_size),
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_data=my_validation_batch_generator,
#                     validation_steps=(num_validation_samples // batch_size),
#                     use_multiprocessing=True,
#                     workers=16,
#                     max_queue_size=32)
# 如果有多个 CPU 核，可以设置 use_multiprocessing=True,即可在 CPU 上并行运行.
# 设置 workers=CPU 核数，用于 batch 数据生成.