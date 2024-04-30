import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pickle


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           #transform=transforms.ToTensor(),
                                           transform=transform,
                                           download = True)
#検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           #transform=transforms.ToTensor(),
                                           transform=transform,
                                           download = True)

print("train_dataset\n", train_dataset)
print("\ntest_dataset\n", test_dataset)


#batch_size = 256
batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

input_size = 28*28
hidden1_size = 1024
hidden2_size = 512
output_size = 10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device : ", device)
model = Net().to(device)
print(model)

# 損失関数  criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
criterion = nn.CrossEntropyLoss()

# 最適化法の指定  optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_model(model, train_loader, criterion, optimizer, device='cpu'):

    train_loss = 0.0  #trainの損失用の変数を定義
    num_train = 0     #学習回数の記録用の変数を定義 

    # モデルを学習モードに変換
    model.train()
    
    # データの分割数分繰り返す
    # バッチサイズ分のデータで1回パラメータを修正する
    for i, (images, labels) in enumerate(train_loader):

        # batch数をカウント
        num_train += len(labels)

        images, labels = images.to(device), labels.to(device)

        # 勾配を初期化
        optimizer.zero_grad()

        # １推論(順伝播)
        outputs = model(images)
        
        # ２損失の算出
        loss = criterion(outputs, labels)

        # ３勾配計算
        loss.backward()

        # ４パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()
    
    # lossの平均値を取る
    train_loss = train_loss / num_train

    return train_loss

def test_model(model, test_loader, criterion, optimizer, device='cpu'):

    test_loss = 0.0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        
        # lossの平均値を取る
        test_loss = test_loss / num_test
    return test_loss

def learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device='cpu'):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss = test_model(model, test_loader, criterion, optimizer, device=device)
        
        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    
    return train_loss_list, test_loss_list

num_epochs = 30
train_loss_list, test_loss_list = learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)

torch.save(model.state_dict(), '/home/oohara/Projects/model_new.pth')

tr = open('/home/oohara/Projects/train.txt','wb')
pickle.dump(train_loss_list,tr)
te = open('/home/oohara/Projects/test.txt','wb')
pickle.dump(test_loss_list,te)
