import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy
import mnist
from torch.autograd import Variable
import math


class Reshape(nn.Module):  # 自己重写的一个nn.Module的子类，用于在Sequential容器中对张量进行Reshape()操作
  def __init__(self, *args):
    super(Reshape, self).__init__()
    self.shape = args

  def forward(self, x):
    x = x.view(self.shape)
    return x
    # print("before reshape")
    # print(x.shape)
    # x = x.view(self.shape)
    # print("after reshape")
    # print(x.shape)
    # return x


class MnistDEMO(object):

  def __init__(self):
    self.fullset = mnist.train_images()  # 返回一个形状为[60000,28,28]的numpy数组,第一个索引代表图片，后两个指定28x28图片上的像素点
    self.fullset = self.fullset - numpy.mean(numpy.mean(self.fullset[:, :, :]))  # 为张量中的每一个元素都减去平均值
    self.testset = mnist.test_images()  # 返回一个形状为[10000,28,28]的numpy数组,第一个索引代表图片，后两个指定28x28图片上的像素点
    self.fullset_label = mnist.train_labels()  # 返回一个形状为[60000]的numpy数组,代表对应图片的label
    self.testset_label = mnist.test_labels()  # 返回一个形状为[10000]的numpy数组,代表对应图片的label
    self.trainset = self.fullset[0:50000, :, :]  # 从60000张图片中选取前50000张做训练集
    self.trainset_label = self.fullset_label[0:50000]  # 从60000张图片中选取前50000个标签做训练集的label
    self.validationset = self.fullset[50000:60000, :, :]  # 从60000张图片中选取后10000个图片做验证集
    self.validation_label = self.fullset[50000:60000]  # 从60000张图片中选取后10000个标签做验证集的label
    # 转tensor
    self.testset = torch.from_numpy(self.testset)
    self.testset_label = torch.from_numpy(self.testset_label.astype(numpy.int64))
    self.trainset = torch.from_numpy(self.trainset)
    self.trainset_label = torch.from_numpy(self.trainset_label.astype(numpy.int64))
    # 训练参数
    self.learningRate = 1e-2
    self.learningRateDecay = 1e-4
    self.weightDecay = 1e-3
    self.momentum = 1e-4
    self.batch_size = 200
    # 先转换成 torch 能识别的 TensorDataset
    self.torch_trainset = torch.utils.data.TensorDataset(self.trainset, self.trainset_label)
    # 把 TensorDataset 放入 DataLoader
    self.train_loader = torch.utils.data.DataLoader(
      dataset=self.torch_trainset,
      batch_size=self.batch_size,
      shuffle=True,  # 打乱数据
      num_workers=2,  # 多线程来读数据
    )
    # 先转换成 torch 能识别的 TensorDataset
    self.torch_testset = torch.utils.data.TensorDataset(self.testset, self.testset_label)
    # 把 TensorDataset 放入 DataLoader
    self.test_loader = torch.utils.data.DataLoader(
      dataset=self.torch_testset,
      batch_size=self.batch_size,
      shuffle=True,  # 打乱数据
      num_workers=2,  # 多线程来读数据
    )
    # 定义代价函数，使用交叉熵验证
    self.criterion = nn.CrossEntropyLoss(reduction='sum')
    # 下面开始搭建神经网络模型
    self.model = nn.Sequential(  # 创建一个时序容器
      nn.Conv2d(1, 16, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(16, 32, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(2),
      Reshape(-1, 32 * 7 * 7),
      nn.Linear(32 * 7 * 7, 10)
      # nn.Softmax(dim=1)
    )
    # 使用GPU计算
    self.model = self.model.cuda()
    # 直接定义优化器，而不是调用backward
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    # 调用参数初始化方法初始化网络参数
    self.model.apply(self.weight_init)
    # 训练函数

  # 参数值初始化
  def weight_init(self, m):
    if isinstance(m, nn.Conv2d):
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
      m.weigth.data.fill_(1)
      m.bias.data.zero_()

  def train(self):
    for epoch in range(1000):  # 训练整套数据 1000 次
      for step, (batch_data, batch_label) in enumerate(self.train_loader):  # 每一步 loader 释放一小批数据用来学习
        batch_data = batch_data.float()
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_data = batch_data.view(-1, 1, 28, 28)
        self.optimizer.zero_grad()
        outputs = self.model(torch.autograd.variable(batch_data))
        loss = self.criterion(outputs, batch_label)
        loss.backward()
        self.optimizer.step()
        if step % 200 == 0:
          test_output = self.model(batch_data[0:100])
          pred = torch.max(test_output, 1)[1].data.squeeze()
          accuracy = sum(pred == batch_label[0:100])
          print('Epoch:', epoch, '|Step:', step, '|train loss:%.4f' % loss.data, '|test accuracy:%.4f' % accuracy)
      if epoch % 20 == 0:
        self.eval()

  def eval(self):
    hit = 0
    for step, (batch_testdata, batch_testlabel) in enumerate(self.test_loader):
      batch_testdata = batch_testdata.cuda().float().view(-1, 1, 28, 28)
      batch_testlabel = batch_testlabel.cuda()
      test_output = self.model(batch_testdata)
      pred = torch.max(test_output, 1)[1].data.squeeze()
      hit += sum(pred == batch_testlabel).__float__()
    accuracy = hit / 100
    print('到目前为止模型的准确率为:%.4f ' % accuracy)


if __name__ == '__main__':
  model = MnistDEMO()
  model.train()
  model.eval()
