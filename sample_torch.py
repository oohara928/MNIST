import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle

trans = torchvision.transforms.ToTensor()
t_gray = [0,1,2,3,4,5,6,7,8,9]
from PIL import Image
im_gray0 = np.array(Image.open("/home/oohara/Documents/0.jpeg").convert('L'))
im_gray1 = np.array(Image.open("/home/oohara/Documents/1.jpeg").convert('L'))
im_gray2 = np.array(Image.open("/home/oohara/Documents/2.jpeg").convert('L'))
im_gray3 = np.array(Image.open("/home/oohara/Documents/3.jpeg").convert('L'))
im_gray4 = np.array(Image.open("/home/oohara/Documents/4.jpeg").convert('L'))
im_gray5 = np.array(Image.open("/home/oohara/Documents/5.jpeg").convert('L'))
im_gray6 = np.array(Image.open("/home/oohara/Documents/6.jpeg").convert('L'))
im_gray7 = np.array(Image.open("/home/oohara/Documents/7.jpeg").convert('L'))
im_gray8 = np.array(Image.open("/home/oohara/Documents/8.jpeg").convert('L'))
im_gray9 = np.array(Image.open("/home/oohara/Documents/9.jpeg").convert('L'))
for i in range(10):
 if i == 0:
  t_gray[i] = trans(im_gray0)
 if i == 1:
  t_gray[i] = trans(im_gray1)
 if i == 2:
  t_gray[i] = trans(im_gray2)
 if i == 3:
  t_gray[i] = trans(im_gray3)
 if i == 4:
  t_gray[i] = trans(im_gray4)
 if i == 5:
  t_gray[i] = trans(im_gray5)
 if i == 6:
  t_gray[i] = trans(im_gray6)
 if i == 7:
  t_gray[i] = trans(im_gray7)
 if i == 8:
  t_gray[i] = trans(im_gray8)
 if i == 9:
  t_gray[i] = trans(im_gray9)
print(t_gray[i])


class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x): # x : 入力
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        y = self.fc3(z2)
        return y

input_size = 28*28
hidden1_size = 1024
hidden2_size = 512
output_size = 10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device : ", device)
model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device)
model.load_state_dict(torch.load('/home/oohara/Projects/model.pth'))
print(model)

tr = open('./train.txt','rb')
train_loss_list = pickle.load(tr)
te = open('./test.txt','rb')
test_loss_list = pickle.load(te)

plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(20, 10))
for i in range(10):
    image, label = t_gray[i],i
    image = image.view(-1, 28*28).to(device)

    # 推論
    prediction_label = torch.argmax(model(image))

    ax = plt.subplot(1, 10, i+1)

    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
plt.show()
