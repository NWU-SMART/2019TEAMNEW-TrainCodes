'''
读取数据，并将数据分成训练集和测试集,未完成
'''

# 导入需要的包
import numpy as np
# 生成随机种子并固定值
'''
传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，
如果使用相同的seed()值，则每次生成的随机数都相同
'''
import random
random.seed(1)
import math
import pickle

# 由于Instagram.data数据是将四种数据地址写在一行
# 如下图所示，第一个空TAB之前为post的id 31
# 后面为文本标签的索引 1 56 1342 180对应于vocabulary中的值
# 第二个TAB后为post对应的hashtag的索引 2477 353 3817 对应于tags中的值
# 12858对应用户的id，可以有重复性（一个用户可以发布多个帖子）
# 31	1 56 1342 180	2477 353 3817	12858

# 对于上述数据首先需要将代表不同含义的数据分离

def data_split():
    # 以只读的方式打开Instagram.data
    data_read = open("shiyan.data", "r")
    # data_read = open("Instagram.data", "r")
    # 为所有的post数据建立一个空数组
    post_whole = []
    # 依次读取每一行数据
    for line in data_read.readlines():
    #将数据分割\t代表Tab
        post_alone = line.strip().split("\t")
        # 通过tag分割后成为一个列数为4的数组，每个数组对应各个
        post_text = [int(x) for x in post_alone[1].split(" ")]
        post_tag = [int(x) for x in post_alone[2].split(" ")]
        # # 将数据重组
        post_whole.append((np.array(post_text),np.array(post_tag),post_alone[0],post_alone[3]))
    print('load data over')
    #将数据分离

    # 帖子的id
    post_id = {}
    for id in post_whole:
        post_id[id[2]] = id
    # print(post_id)

    #  由于用户的id可能会出现重复，因此每次写入用户的id时需要考虑是否已经写入了
    #  用户的id,跟随着发布的帖子
    use_id = {}
    for id in post_whole:
        if id[3] not in use_id.keys():
            use_id[id[3]] = []
        use_id[id[3]].append(id[2])
    # print(use_id)
# 这里没有考虑用户的历史帖子，所以没有对用户的历史帖子进行提取


#     正式开始划分数据集
#     post的id划分
    train_postid = []
    test_postid = []
    for id in post_id.keys():
        #随机从数据中选1/10的数据用来测试，剩下的用来训练
        post_sample = random.sample(post_id[id], int(math.ceil(len(post_id[id]) / 10)))
        # 将随机选择的1/10数据当初测试数据
        test_postid.extend(post_sample)
        # 测试数据剩下的就是训练数据
        train_postid.extend([i for i in post_id[id] if i not in post_sample])
# #       user的id划分
#     use_id_sample = []
#     for id in use_id.keys():
#         use_id_sample.extend(use_id[id])
#     print(len(train_postid), len(test_postid), len(use_id_sample), len(post_whole))


if __name__ == "__main__":
    data_split()


