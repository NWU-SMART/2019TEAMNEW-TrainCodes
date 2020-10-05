# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/26 002620:57
# 文件名称：anchors
# 开发工具：PyCharm
import numpy as np

'''
在一副图片中找8732个先验框，并且表示出来

'''


class PriorBox():
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):
        self.waxis = 1
        self.haxis = 0
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]  # 最开始尺寸只有1.0
        # 如果最大值存在
        if max_size:
            if max_size < min_size:  # 如果最大值比最小值还小，就报错
                raise Exception('max_size must be greater than min_size')
            self.aspect_ratios.append(1.0)  # self.aspect_ratios现在是[1.0,1.0]两个1.0

        if aspect_ratios:  # 如果输入进来的尺寸存在
            for ar in aspect_ratios:  # 得到输入进来的每个尺寸
                if ar not in self.aspect_ratios:  # 如果输入进来的尺寸不存在self.aspect_ratios
                    self.aspect_ratios.append(ar)  # 就将输入尺寸添加到self.aspect_ratios中
                    self.aspect_ratios.append(1.0 / ar)  # 将倒数也添加进去
        self.variances = np.array(variances)
        self.clip = True

    def call(self, input_shape, mask=None):
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        # 获取输入进来图片的宽和高
        # 300,300
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # 获得先验框的宽和高
        box_widths = []
        box_heights = []

        # 下面for循环是把预测框的长宽得到
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:  # 第一个比列，最小的正方形
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:  # 最大正方形
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        box_widths = 0.5 * np.array(box_widths)  # 比例缩小一半
        box_heights = 0.5 * np.array(box_heights)  # 比例缩小一半

        # 跳的部署，如对于
        step_x = img_width / layer_width  # 步数就是表示 300是当前尺度的倍数
        step_y = img_height / layer_height

        # 类似np.linspace(1,7,3)=[1,4,7]，此处得到所有先验框的x坐标，(相当于把重复的删去了)
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)  # 开始数据是0.5 * step_x，结束数据是img_width - 0.5 * step_x，长度为layer_width，中间平分

        # 此处得所有先验框的y坐标,和上面类似，因为两个宽和高一样，完全可以复制一份,如：liny = linx.copy()
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        # 此处将linx复制len(liny)行，每行都一样。将liny复制len(linx)列，每列都一样
        centers_x, centers_y = np.meshgrid(linx, liny)

        # 计算网络中心
        centers_x = centers_x.reshape(-1, 1)

        centers_y = centers_y.reshape(-1, 1)

        # 每个网格（cell）有num_priors个先验框
        num_priors = len(self.aspect_ratios)

        # 将中心（centers_x和centers_y进行合并）,priors_boxes[0][0]和priors_boxes[0][1]代表第一个先验框的中心坐标
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        # 左上角x
        prior_boxes[:, ::4] -= box_widths
        # 左上角y
        prior_boxes[:, 1::4] -= box_heights
        # 右下角x
        prior_boxes[:, 2::4] += box_widths
        # 右下角y
        prior_boxes[:, 3::4] += box_heights

        # 变成小数的形式
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        # 将数值定义到0-1之间
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0), 1)

        num_boxes = len(prior_boxes)  # 先验框的个数


        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)


        return prior_boxes  # shape(num_boxes,8),前四个是先验框的坐标（左上，右下，但是是比列，需要*img_size才是真实坐标），后四个是缩放比例（我是这么认为的）


def get_anchors(img_size=(300, 300)):
    net = {}
    # 第一个尺度的先验框
    priorbox = PriorBox(img_size, 30.0, max_size=60.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_box_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox.call([38, 38])

    # 第二个尺度的先验框
    priorbox = PriorBox(img_size, 60.0, max_size=111.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox.call([19, 19])

    # 第三个尺度的先验框
    priorbox = PriorBox(img_size, 111.0, max_size=162.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox.call([10, 10])

    # 第四个尺度的先验框
    priorbox = PriorBox(img_size, 162.0, max_size=213.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox.call([5, 5])

    # 第五个尺度的先验框
    priorbox = PriorBox(img_size, 213.0, max_size=264.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox.call([3, 3])

    # 第六个尺度的先验框
    priorbox = PriorBox(img_size, 264, max_size=315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
    net['pool6_mbox_priorbox'] = priorbox.call([1, 1])


    net['mbox_priorbox'] = np.concatenate([
        net['conv4_3_norm_mbox_priorbox'],
        net['fc7_mbox_priorbox'],
        net['conv6_2_mbox_priorbox'],
        net['conv7_2_mbox_priorbox'],
        net['conv8_2_mbox_priorbox'],
        net['pool6_mbox_priorbox']
    ],axis=0)
    # shape(8732,8),前四个为坐标（比列，坐标处于image—_size），[0.03043413 0.         0.10114481 0.08386857 0.1        0.1     0.2        0.2       ]，
    return net['mbox_priorbox']
