import lib.shopping_data as shopping_data
import lib.chinese_vec as chinese_vec  # 读取中文词向量工具
import keras
from keras.utils import pad_sequences  # 数据对齐
from keras.models import Sequential  # 堆叠神经网络序列的载体
from keras.layers import Dense, Embedding  # 引入全连接层，一层神经网络；引入嵌入层
from keras.layers import Flatten  # 数组平铺
from keras.layers import LSTM  # 导入LSTM
import numpy as np

x_train, y_train, x_test, y_test = shopping_data.load_data()
# 第一步，把数据集中所有的文本转化为词典的索引值，程序去遍历语料中的句子，那如果是中文，就进行分词
voca_len, word_index = shopping_data.createWordIndex(x_train, x_test)
# voca_len为这个词典的词汇数量, word_index为训练集和测试集全部预料的词典
print(word_index)
print('词典总词数：', voca_len)

# 第二步，将每句话转化为索引向量
max_len = 25
x_train_index = shopping_data.word2Index(x_train, word_index, max_len)
x_test_index = shopping_data.word2Index(x_test, word_index, max_len)

# 第四步 构建词嵌入矩阵
word_vecs = chinese_vec.load_word_vecs()
embedding_matrix = np.zeros((voca_len, 300))  # 把嵌入层的嵌入矩阵替换为0矩阵，（语调数量，维度数量300）
# 从预处理中词库找到这个词的索引
for word, i in word_index.items():
    embedding_vector = word_vecs.get(word)
    if embedding_vector is not None:  # 有可能这个词不在词库中
        embedding_matrix[i] = embedding_vector  # 如果有，替换这一行的词向量

# 神经网络模型
model = Sequential()
model.add(
    Embedding(
        trainable=False,  # 是否可训练：是否让这一层在训练的时候更新参数
        weights=[embedding_matrix],
        input_dim=voca_len,  # 输入维度，所有词汇的数量
        output_dim=300,  # 输出维度，每个词向量的特征维度
        input_length=max_len  # 序列长度，每句话长度为25的对齐
    )
)

# LSTM神经网络
model.add(LSTM(
    128,  # 输出数据的维度
    return_sequences=True  # 每一个都输出结果
))  # 前面一层需要每一步输出结果，作为下一层的输入
model.add(LSTM(128))

# 二分类问题，使用sigmoid激活函数
model.add(Dense(1, activation='sigmoid'))

# 配置模型
model.compile(
    loss='binary_crossentropy',  # 适用于二分类问题的交叉熵代价函数
    optimizer='adam',  # adam是一种使用动量的自适应优化器，比普通的sgd优化器更快
    metrics=['accuracy']
)

# loss（损失函数、代价函数）：binary_crossentropy适用于二分类问题的交叉熵代价函数
# optimizer（优化器）：adam是一种使用动量的自适应优化器，比普通的sgd优化器更快
# metrics（评估标准）：accuracy（准确度）；

# 训练数据fit， batch_size送入批次的数据，显卡越好，可以送的数据越多，只能用CPU训练，batch_size就设置小一点
model.fit(x_train_index, y_train, batch_size=1024, epochs=200)
# 用来测试集上的评估
score, acc = model.evaluate(x_test_index, y_test)
print('Test score:', score)
print('Test accuracy:', acc)
# 保存模型
model.save(f'{acc:.4f}_model.h5')
