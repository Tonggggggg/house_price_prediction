import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import csv

train_data = pd.read_csv('train_afterchange.csv')
test_data = pd.read_csv('test_afterchange.csv')
X_train = train_data.iloc[:1058, 1:-1]
X_valid = train_data.iloc[1058:, 1:-1]
Y_train = train_data.iloc[:1058, -1]
Y_valid = train_data.iloc[1058:, -1]
X_test = test_data.iloc[:, 1:]
X_ = X_train.as_matrix()
Y_ = np.array([Y_train])
X_valid = X_valid.as_matrix()
Y_valid = np.array([Y_valid])
X_test_ = X_test.as_matrix()
# print(X_)
# print(Y_)
sx = np.column_stack([X_, np.zeros((1058, 1))])
sx_valid = np.column_stack([X_valid, np.zeros((400, 1))])
sx_test = np.column_stack([X_test_, np.zeros((1459, 1))])


# 变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积处理 变厚过程
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pool 长宽缩小一倍
# def max_pool_2x2(x):
#     # stride [1, x_movement, y_movement, 1]
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 400])  # 原始数据的维度：400
ys = tf.placeholder(tf.float32, [None, 1])  # 输出数据为维度：1

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 20, 20, 1])  # 原始数据400变成二维图片20*20

## conv1 layer ##第一卷积层
W_conv1 = weight_variable([2, 2, 1, 32])  # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 20x20x32，长宽不变，高度为32的三维图像

## conv2 layer ##第二卷积层
W_conv2 = weight_variable([2, 2, 32, 64])  # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 输入第一层的处理结果 输出shape 20*20*64

## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([20 * 20 * 64, 512])  # 20*20 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_conv2, [-1, 20 * 20 * 64])  # 把20*20，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 把数组中扔掉比例为keep_prob的元素

## fc2 layer ## full connection
W_fc2 = weight_variable([512, 1])  # 512长的一维数组压缩为长度为1的数组
b_fc2 = bias_variable([1])  # 偏置

# 最后的计算结果
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 0.01学习效率,minimize(loss)减小loss误差
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
Y_=Y_.T
Y_valid = Y_valid.T

sess.run(tf.global_variables_initializer())
loss_list = []
valid_loss_list = []
for i in range(400):
    sess.run(train_step, feed_dict={xs: sx, ys: Y_, keep_prob: 0.7})
    sess.run(train_step, feed_dict={xs: sx_valid, ys: Y_valid, keep_prob: 0.7})
    loss = sess.run(cross_entropy, feed_dict={xs: sx, ys: Y_, keep_prob: 1.0})
    valid_loss = sess.run(cross_entropy, feed_dict={xs: sx_valid, ys: Y_valid, keep_prob: 1.0})
    loss_list.append(loss)
    valid_loss_list.append(valid_loss)
    print(i, 'loss=', loss, 'valid_loss=', valid_loss)

print(len(loss_list), len(valid_loss_list))

print(prediction_value )
fig1 = plt.figure(figsize=(20, 3))
axes = fig1.add_subplot(1, 1, 1)
line1, = axes.plot(range(len(loss_list[35:])),loss_list[35:], 'r', label=u'train_loss', linewidth=2)
line2, = axes.plot(range(len(valid_loss_list[35:])), valid_loss_list[35:], 'g', label=u'valid_loss')
axes.grid()
fig1.tight_layout()
plt.legend(handles=[line1, line2])
plt.title('loss')
plt.show()
new_prediction_value=np.expm1(prediction_value)
sess.close()

#写入csv文件
Id_list=[i for i in range(1461,2920)]
print (len(Id_list))
price_list=[]
for i in range(0,1459):
    new_list=[]
    new_list=[Id_list[i],new_prediction_value[i][0]]
    price_list.append(new_list)
print(price_list)


fileHeader = ["Id", "SalePrice"]
csvFile = open("sample_submission.csv", "w",newline='')
writer = csv.writer(csvFile)
writer.writerow(fileHeader)
writer.writerows(price_list)
csvFile.close()