# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import MLP                                          # 导入fc.py中的两层全连接层
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------

# MFB模型
class MFB(nn.Module):
    def __init__(self, __C, img_feat_size, ques_feat_size, is_first):
        super(MFB, self).__init__()
        self.__C = __C
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, __C.MFB_K * __C.MFB_O)   # 输入：图像大小  MFB_K = 5,MFB_O = 1000，输出：5000
        self.proj_q = nn.Linear(ques_feat_size, __C.MFB_K * __C.MFB_O)  # 输入：问题大小  MFB_K = 5,MFB_O = 1000，输出：5000
        self.dropout = nn.Dropout(__C.DROPOUT_R)                        # dropout层
        self.pool = nn.AvgPool1d(__C.MFB_K, stride=__C.MFB_K)           # 平均池化，核大小5，步长5  大小变为1/5

    def forward(self, img_feat, ques_feat, exp_in=1):                  # 前向传播
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)        若MFB_K = 5,MFB_O = 1000,则K*O=5000
        '''
        batch_size = img_feat.shape[0]                                                          # 批大小等于输入图像 N
        img_feat = self.proj_i(img_feat)                                                        # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)                                                      # (N, 1, K*O)

        exp_out = img_feat * ques_feat                     # 图像和文本 点乘(Eltwise Multilication)         (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)    # dropout层 (N, C, K*O)
        z = self.pool(exp_out) * self.__C.MFB_K            # 平均池化（论文的sum pooling）                  (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z)) # z先经过relu激活再经过开方，对-z做同样操作 再做差（论文l2正则？）
        z = F.normalize(z.view(batch_size, -1))            # 归一化（论文power normalization）              (N, C*O)
        z = z.view(batch_size, -1, self.__C.MFB_O)         # (N, C, O)   (N,C,1000)
        return z, exp_out


# 问题attention模型
class QAtt(nn.Module):
    def __init__(self, __C):
        super(QAtt, self).__init__()
        self.__C = __C
        self.mlp = MLP(                                # 使用MLP(两层全连接)
            in_size=__C.LSTM_OUT_SIZE,                 # 输入大小=1024
            mid_size=__C.HIDDEN_SIZE,                  # 中间大小=512
            out_size=__C.Q_GLIMPSES,                   # 输出大小=2
            dropout_r=__C.DROPOUT_R,                   # dropout rate =0.1
            use_relu=True                              # 使用relu激活函数
        )

    def forward(self, ques_feat):                     # 前向传播
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # 经过mlp(两层linear层)    (N, T, Q_GLIMPSES) （N,T,2）
        qatt_maps = F.softmax(qatt_maps, dim=1)         # 经过softmax(论文下面蓝色部分已实现) (N, T, Q_GLIMPSES)

        qatt_feat_list = []                             # 定义一个空列表先
        for i in range(self.__C.Q_GLIMPSES):            # 0-2
            mask = qatt_maps[:, :, i:i + 1]             # 列表分片：前两列不动，取出i-i+1 这维 所以大小变为 (N, T, 1)
            mask = mask * ques_feat                     # 将取出来的和ques_feat做点乘  大小变为(N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # dim=1 横向(压缩)求和 (N, LSTM_OUT_SIZE)          (N,1024)
            qatt_feat_list.append(mask)                 # 放入列表中
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # 拼接     大小变为 (N, LSTM_OUT_SIZE*Q_GLIMPSES) （N,1024*2）

        return qatt_feat

# 图像attention模型
class IAtt(nn.Module):
    def __init__(self, __C, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.__C = __C
        self.dropout = nn.Dropout(__C.DROPOUT_R)                       # dropout rate =0.1
        self.mfb = MFB(__C, img_feat_size, ques_att_feat_size, True)   # 将提取出来的图像特征和问题attention进行MFB融合
        self.mlp = MLP(                                                # 经过mlp(两层linear层)
            in_size=__C.MFB_O,                                         # 输入大小=1000
            mid_size=__C.HIDDEN_SIZE,                                  # 中间大小=512
            out_size=__C.I_GLIMPSES,                                   # 输出大小=2
            dropout_r=__C.DROPOUT_R,                                   # dropout rate =0.1
            use_relu=True                                              # 使用relu激活函数
        )

    def forward(self, img_feat, ques_att_feat):                       # 前向传播
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # unsqueeze（1）是增添第1个维度为1，以插入的形式填充  变为(N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)               # dropout
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)   z的大小(N,C,1000)

        iatt_maps = self.mlp(z)                         # 经过mlp(两层linear层)    (N, C, I_GLIMPSES) (N,C,2)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # 经过softmax(论文上面绿色部分已实现)(N, C, I_GLIMPSES)

        iatt_feat_list = []                             # 定义一个空列表先
        for i in range(self.__C.I_GLIMPSES):            # 0-2
            mask = iatt_maps[:, :, i:i + 1]             # 列表分片：前两列不动，取出i-i+1 这维 所以大小变为 (N, C, 1)
            mask = mask * img_feat                      # 将取出来的和img_feat做点乘  大小变为(N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # dim=1 横向(压缩)求和 (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)                 # 放入列表中
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # # 拼接     大小变为 (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat

# co-attention模型
class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE']          # 获取img_feat_size的大小
        img_att_feat_size = img_feat_size * __C.I_GLIMPSES                    # 获取img_att_feat_size的大小
        ques_att_feat_size = __C.LSTM_OUT_SIZE * __C.Q_GLIMPSES               # 获取ques_att_feat_size的大小

        self.q_att = QAtt(__C)                                                # 问题attention
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)             # 图像attention

        if self.__C.HIGH_ORDER:  # MFH： 如果是高阶的，用两个MFB融合
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB ：否则只使用一次
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # 问题大小： (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # 融合后的大小： (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        if self.__C.HIGH_ORDER:  # MFH ：如果是高阶的，用两个MFB融合                   # unsqueeze（1）是增添第1个维度为1，以插入的形式填充
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # 将混合的特征和问题attention再融合 z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # 去掉第1维，然后拼接 (N, 2*O)
        else:  # MFB  ：否则只使用一次
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # 去掉第1维 (N, O)

        return z
