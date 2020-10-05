# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/20 002022:02
# 文件名称：SSD
# 开发工具：PyCharm

from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from nets.VGG16 import VGG16

from nets.SSD_layers import Normalize, PriorBox

'''
###############################（三）#############################

SSD300的模型建立，
预测结果为shape-->(k,8732,33),
k表示图片数量，
8732，表示先验框数量
33表示对于每一个先验框，前4位表示预测偏移，中间21位表示置信度，后8位表示左上（x,y）,右下(x,y).和比列
'''

def SSD300(input_shape, num_classes=21):
    # 300,300,3
    input_tensor = Input(shape=(input_shape))
    img_size = (input_shape[1], input_shape[0])

    # SSD结构，net字典
    net = VGG16(input_tensor)
    # ---------------------将提取到的主干特征进行处理--------------

    # 对conv4_3进行处理 38,38,512
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 4  # 每个cell对应4个先验框,num_priors*4表示一个先验框对应四个点，4是x，y，h,w的调整
    # 预测框处理 38,38,512->38,38,16
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                          name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    # 展开 38,38,16->23104
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    # 处理置信度 38,38,512->38,38,84
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                           name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    # 38,38,84->121296
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 30.0, max_size=60.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])

    # 对FC7层进行处理19,19,1024
    num_priors = 6
    # 位置处理 19,19,1024->19,19,24
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                 name='fc7_mbox_loc')(net['fc7'])
    # 19,19,24->8664
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # 置信度 19,19,1024->19,19,126
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                  name='fc7_mbox_conf')(net['fc7'])
    # 19,19,126->45486
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 60.0, max_size=111.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])

    # 对conv6_2进行处理5,5,256
    num_priors = 6
    # 位置处理 10,10,256->10,10,24
    net['conv6_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                     name='conv6_2_mbox_loc')(net['conv6_2'])
    # 10,10,24->2400
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # 置信度 10,10,256->10,10,126
    net['conv6_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                      name='conv6_2_mbox_conf')(net['conv6_2'])
    # 10,10,126->12600
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 111.0, max_size=162.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])

    # 对conv7_2进行处理 5,5,256
    num_priors = 6
    # 位置处理 5,5,256->5,5,24
    net['conv7_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                     name='conv7_2_mbox_loc')(net['conv7_2'])
    # 5,5,24->600
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # 置信度 5,5,256->5,5,126
    net['conv7_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                      name='conv7_2_mbox_conf')(net['conv7_2'])
    # 5,5,126->3150
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 162.0, max_size=213.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    # 对conv8_2进行处理 3,3,256
    num_priors = 4
    # 位置处理 3,3,256->3,3,16
    net['conv8_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                     name='conv8_2_mbox_loc')(net['conv8_2'])
    # 3,3,16->144
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # 置信度 3,3,256->3,3,84
    net['conv8_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                      name='conv8_2_mbox_conf')(net['conv8_2'])
    # 3,3,84->756
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 213.0, max_size=264.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    # 对conv9_2处理 1,1,256
    num_priors = 4
    # 位置处理 1,1,256->1,1,16
    net['conv9_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                     name='conv9_2_mbox_loc')(net['conv9_2'])
    # 1,1,16->16
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # 置信度处理 1,1,256->1,1,84
    net['conv9_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                      name='conv9_2_mbox_conf')(net['conv9_2'])
    # 1,1,84->84
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    # 先验框
    priorbox = PriorBox(img_size, 264.0, max_size=315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')
    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    # 8732*4
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['conv4_3_norm_mbox_loc_flat'],  # 23104
                                                            net['fc7_mbox_loc_flat'],  # 8664
                                                            net['conv6_2_mbox_loc_flat'],  # 2400
                                                            net['conv7_2_mbox_loc_flat'],  # 600
                                                            net['conv8_2_mbox_loc_flat'],  # 144
                                                            net['conv9_2_mbox_loc_flat']])  # 16

    # 8732*21
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['conv4_3_norm_mbox_conf_flat'],  # 121296
                                                              net['fc7_mbox_conf_flat'],  # 45486
                                                              net['conv6_2_mbox_conf_flat'],  # 12600
                                                              net['conv7_2_mbox_conf_flat'],  # 3150
                                                              net['conv8_2_mbox_conf_flat'],  # 756
                                                              net['conv9_2_mbox_conf_flat']])  # 84

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([net['conv4_3_norm_mbox_priorbox'],
                                                                      net['fc7_mbox_priorbox'],
                                                                      net['conv6_2_mbox_priorbox'],
                                                                      net['conv7_2_mbox_priorbox'],
                                                                      net['conv8_2_mbox_priorbox'],
                                                                      net['conv9_2_mbox_priorbox']])

    # 8732,4
    net['mbox_loc'] = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])

    # 8732,21
    net['mbox_conf'] = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = Concatenate(axis=2, name='predictions')([net['mbox_loc'],
                                                                  net['mbox_conf'],
                                                                  net['mbox_priorbox']])

    model = Model(net['input'], net['predictions'])
    return model


SSD300(input_shape=(300, 300, 3)).summary()
