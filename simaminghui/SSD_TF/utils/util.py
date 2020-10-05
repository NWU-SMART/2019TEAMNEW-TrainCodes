# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/30 00309:56
# 文件名称：utils
# 开发工具：PyCharm
import warnings

import numpy as np
import tensorflow as tf

from utils.anchors import get_anchors


class BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_class = num_classes
        self.priors = priors  # 先验框
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nums_thresh = nms_thresh
        self._top_k = top_k

    def iou(self, box):
        '''
        :param box: 为真实框
        :return: [0.1 0.5 0 0 0.6....0.5] shape=(1,8732)
        '''
        upleft = np.maximum(self.priors[:, :2], box[:2])  # 左上角
        botright = np.minimum(self.priors[:, 2:4], box[2:])  # 右下角

        wh = botright - upleft
        wh = np.maximum(wh, 0)
        # 重合部分面积
        inter = wh[:, 0] * wh[:, 1]

        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        iou = inter / (area_true + area_gt - inter)
        return iou

    def encode_box(self, box):
        '''
        :param box: 表示一个真实框
        :return:    shape = （8732*5）一维向量，只有对应的先验框（预测框）上才有值，其他均为0
        '''
        return_encode_info = np.zeros((self.num_priors, 5))  # 全为0，shape为（8732,5）的矩阵，然后在里面添加信息，最后返回
        iou = self.iou(box)  # 得到当前真实框与所有先验框的iou，如[0.1 0.5 0 0 0.6....0.5] shape=(1,8732)
        assign_mask = iou > self.overlap_threshold  # 得到符合条件的先验框（条件是与当前真实框的iou大于overlap_threshold），得到一系列[False False False  True  True  True..],shape=(1,8732)

        if not assign_mask.any():  # 如果assign_mask没有true，即8732个先验框中没有一个与当前的真实框的iou大于overlap_threshold
            assign_mask[np.argmax(iou)] = True  # 那么选取先验证中与当前真实框iou最大的值，此时这个先验框与真实框的iou小于overlap_threshold

        return_encode_info[:, -1][assign_mask] = iou[assign_mask]  # 筛选后的先验框（后面叫做预测框）,最后一列得到相对应的iou

        assigned_priors = self.priors[assign_mask]  # 得到相应的先验框（此时为预测框了）

        assigned_priors_center_xy = (assigned_priors[:, :2] + assigned_priors[:, 2:4]) * 0.5  # 得到预测框的中心xy
        assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]  # 得到预测框的wh
        box_center_xy = (box[:2] + box[2:]) * 0.5  # 得到当前真实框的中心xy
        box_wh = box[2:] - box[:2]  # 得到当前真实框的wh

        return_encode_info[:, :2][assign_mask] = (box_center_xy - assigned_priors_center_xy) / assigned_priors_wh
        return_encode_info[:, :2][assign_mask] /= assigned_priors[:, -4:-2]  # 前两个表示x和y的偏移

        # 此时return_encode_info的shape=(8732,5)，8732表示每个先验框，5表示中心坐标xy和宽高wh的偏移和iou
        return_encode_info[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh) / assigned_priors[:,
                                                                                        -2:]  # 后两个表示wh的偏移

        return return_encode_info  # ravel将多维数组转为1维，但是不明白为什么要转

    def assign_boxes(self, boxes):  #
        assignment = np.zeros(
            (self.num_priors, 4 + self.num_class + 8))  # 33中，前四个是偏移，中间21是置信度，接下来4个是先验框的坐标，最后4个是比列，为了扩大损失函数

        assignment[:, 4] = 1.0
        if len(boxes) == 0:  # 如果一副图片中没有真实框（全是背景）
            return assignment

        # 对每一个真实框进行iou计算 下面想相当于for循环
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        # 每一个真实框编码后的值，和iou,encoded_boxes-->shape(f,8732,5)，f为真实框的个数
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合度最大的先验框，并且获取这个先验框的下标，encoded_boxes[:,:,-1]-->shape(f,8732)，对于一行，8732个。每个表示对于当前真实框的iou。
        best_iou = encoded_boxes[:, :, -1].max(
            axis=0)  # best_iou类似[[0] [0] [0.6] [0.7]..],0表示当前先验框没有拼配任何真实框，0.6或者0.7表示当前先验框匹配到的真实框最大的iou为0.6或0.7，shape=(8732,1)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)  # 类似[0 1 0 2 3 4...]其中数字最大为f-1(索引)。shape-->(8732,1),表示某个先验框匹配到的真实框的索引，是真实框的索引
        best_iou_mask = best_iou > 0  # 类似[[Flase] [Flase] [True]..],Flase表示当前先验框没有和任何真实框匹配，也就是背景。True表示先验框有匹配到的真实框，shape=（8732,1）
        best_iou_idx = best_iou_idx[best_iou_mask]  # 得到真实框的索引，类似=[2 0 3 4] shape=(w,1),w表示有用先验框(正样本)的个数

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]  # 得到有效的先验框，即如果先验框对于所有的真实框iou<0.5,舍去，留下有用的先验框
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor  # 监视的值
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':  # 如果是val_locc,则选取这个
            self.monitor_op = np.less  # np.less相当于<
            self.best = np.Inf  # 表示无穷大,只是一个符号，比任何数都大

        elif mode == 'max':  # 用于检测准确数,acc,越大越好
            self.monitor_op = np.greater
            self.best = -np.Inf  # 负无穷

        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):  # 如果监视的还是acc
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_best_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))

            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)