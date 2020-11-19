import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

from DNN.Example_CNN import CNN

train_dataset = dsets.MNIST(root='./datasets',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./datasets',
                           train=False,
                           transform=transforms.ToTensor())
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
model = CNN(in_c_1=1,out_c_1=25,in_c_2=25,out_c_2=50)
criterion = nn.CrossEntropyLoss()


learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # 梯度置零
        optimizer.zero_grad()

        # 计算输出
        outputs = model(images)

        # 计算损失，内部会自动softmax然后进行Crossentropy
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # 计算准确度
            correct = 0
            total = 0
            for images, labels in test_loader:
                #images = Variable(images.view(-1, 28 * 28))

                # 获得输出，输出的大小为(batch_size,10)
                outputs = model(images)

                # 获得预测值，输出的大小为(batch_size,1)
                _, predicted = torch.max(outputs.data, 1)

                # labels的size是(100,)
                total += labels.size(0)

                # 返回的是预测值和标签值相等的个数
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
torch.save(model.state_dict(), './pretrain/cnn.pkl')
# 计算准确度
correct = 0
total = 0
for images, labels in test_loader:
    #images = Variable(images.view(-1, 28 * 28))

    # 获得输出，输出的大小为(batch_size,10)
    outputs = model(images)

    # 获得预测值，输出的大小为(batch_size,1)
    _, predicted = torch.max(outputs.data, 1)

    # labels的size是(100,)
    total += labels.size(0)

    # 返回的是预测值和标签值相等的个数
    correct += (predicted == labels).sum()

accuracy = 100 * correct / total

# Print Loss
print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))