# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.models.mfb.mfb import CoAtt
from openvqa.models.mfb.adapter import Adapter
import torch
import torch.nn as nn


# -------------------------------------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------------------------------------

# 总网络结构
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.adapter = Adapter(__C)

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE       # 300
        )


        if __C.USE_GLOVE:                                      # 载入GloVe权重：外部预先训练好的单词嵌入模型的方法
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(                                   # 一层LSTM
            input_size=__C.WORD_EMBED_SIZE,                    # 300
            hidden_size=__C.LSTM_OUT_SIZE,                     # 1024
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)              # dropout
        self.dropout_lstm = nn.Dropout(__C.DROPOUT_R)         # dropout
        self.backbone = CoAtt(__C)                            # 调用co-attention

        if __C.HIGH_ORDER:      # MFH                         # 最后答案输出
            self.proj = nn.Linear(2*__C.MFB_O, answer_size)
        else:                   # MFB
            self.proj = nn.Linear(__C.MFB_O, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):    # 前向传播

        img_feat, _ = self.adapter(frcn_feat, grid_feat, bbox_feat)  # 获取 img_feat (N, C, FRCN_FEAT_SIZE)

        # Pre-process Language Feature：预处理语言特征
        ques_feat = self.embedding(ques_ix)         # (N, T, WORD_EMBED_SIZE)
        ques_feat = self.dropout(ques_feat)         # dropout
        ques_feat, _ = self.lstm(ques_feat)         # (N, T, LSTM_OUT_SIZE)
        ques_feat = self.dropout_lstm(ques_feat)    # dropout

        z = self.backbone(img_feat, ques_feat)     # co-attention :MFH:(N, 2*O) / MFB:(N, O)
        proj_feat = self.proj(z)                   # 最后答案输出  (N, answer_size)

        return proj_feat

