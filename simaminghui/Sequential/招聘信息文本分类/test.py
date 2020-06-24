# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/23 002321:12
# 文件名称：test
# 开发工具：PyCharm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import sequence

text = 'L LOVE YOU'
# 1.告诉token我要建立一个2000个单词的字典
token = Tokenizer(num_words=7)
# 2.告诉token现在开始读文章（text），并且对文章里面的单词出现过的次数做统计，从高到低依次排列。
token.fit_on_texts(text)
# 运行完这一句token就将所有单词的出现过的次数统计好了1代表出现频率最高的单词，2代表出现频率第二高的单词，3代表.....
# 可以用token.word_index查看单词排列顺序。
print(token.word_index)  # {'l': 1, 'o': 2, 'v': 3, 'e': 4, 'y': 5, 'u': 6}
# 3.告诉token把文章（test_text）里的内容替换成数字
x_text = token.texts_to_sequences(text)
print(x_text)
# 4.告诉token将文章中的每句话单词的长度统一到定长
x_text = sequence.pad_sequences(x_text,maxlen=26)
print(x_text)