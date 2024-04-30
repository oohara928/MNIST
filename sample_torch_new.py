import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.io import read_image
import pickle
import cv2
from PIL import Image

trans = torchvision.transforms.ToTensor()
t_gray = [0,1,2,3,4,5,6,7,8,9]
t_gray[0] = read_image(path='/home/oohara/Documents/0.png')/255.0
t_gray[1] = read_image(path='/home/oohara/Documents/1.png')/255.0
t_gray[2] = read_image(path='/home/oohara/Documents/2.png')/255.0
t_gray[3] = read_image(path='/home/oohara/Documents/3.png')/255.0
t_gray[4] = read_image(path='/home/oohara/Documents/4.png')/255.0
t_gray[5] = read_image(path='/home/oohara/Documents/5.png')/255.0
t_gray[6] = read_image(path='/home/oohara/Documents/6.png')/255.0
t_gray[7] = read_image(path='/home/oohara/Documents/7.png')/255.0
t_gray[8] = read_image(path='/home/oohara/Documents/8_2.png')/255.0
t_gray[9] = read_image(path='/home/oohara/Documents/9_2.png')/255.0

from PIL import Image #画像取り込み　numpy変換
'''im_gray0 = np.array(Image.open("/home/oohara/Documents/0.png"))
im_gray1 = np.array(Image.open("/home/oohara/Documents/1.png"))
im_gray2 = np.array(Image.open("/home/oohara/Documents/2.png"))
im_gray3 = np.array(Image.open("/home/oohara/Documents/3.png"))
im_gray4 = np.array(Image.open("/home/oohara/Documents/4.png"))
im_gray5 = np.array(Image.open("/home/oohara/Documents/5.png"))
im_gray6 = np.array(Image.open("/home/oohara/Documents/6.png"))
im_gray7 = np.array(Image.open("/home/oohara/Documents/7.png"))
im_gray8 = np.array(Image.open("/home/oohara/Documents/8_2.png"))
im_gray9 = np.array(Image.open("/home/oohara/Documents/9_2.png"))

for i in range(10): #バイナリ変換　テンソル変換
 if i == 0:
  th, gray_th = cv2.threshold(im_gray0,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 1:
  th, gray_th = cv2.threshold(im_gray1,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 2:
  th, gray_th = cv2.threshold(im_gray2,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 3:
  th, gray_th = cv2.threshold(im_gray3,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 4:
  th, gray_th = cv2.threshold(im_gray4,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 5:
  th, gray_th = cv2.threshold(im_gray5,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 6:
  th, gray_th = cv2.threshold(im_gray6,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 7:
  th, gray_th = cv2.threshold(im_gray7,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 8:
  th, gray_th = cv2.threshold(im_gray8,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
 if i == 9:
  th, gray_th = cv2.threshold(im_gray9,128,255,cv2.THRESH_BINARY)
  t_gray[i] = trans(gray_th).unsqueeze(0)
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        output = x
        return output

input_size = 28*28
hidden1_size = 1024
hidden2_size = 512
output_size = 10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device : ", device)
#model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device)
model = Net().to(device)
model.load_state_dict(torch.load('/home/oohara/Projects/model_new.pth'))
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
    image = image.to(device)
    # 推論
    prediction_label = torch.argmax(model(image))
    #chk1 = model(image)
    #prediction_label = 6

    ax = plt.subplot(1, 10, i+1)

    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
plt.show()
