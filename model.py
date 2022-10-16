import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
import torchvision.models.vgg as vgg
# 모델은 파이토치에서 제공하는 torchvision.models.vgg 함수 사용

vis = visdom.Visdom()
vis.close(env="main")

# 손실 추적기
def loss_tracker(loss_plot, loss_value, num):
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append')

# gpu가 있으면 gpu 사용, 없으면 cpu 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_iter = iter(trainloader)
images, labels = data_iter.next()
vis.images(images / 2 + 0.5)

for i in range(4):
    print(' '.join('%5s' % classes[labels[i]]))

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']

class vgg(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(vgg, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ) # fc layer
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

vgg16 = vgg(vgg.make_layers(cfg), 10, True).to(device)

a = torch.Tensor(1, 3, 32, 32).to(device)
out = vgg16(a)
print(out)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(vgg16.parameters(), lr=0.005, momentum=0.9)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_trakcer', legend=['loss'], showlegend=True))
print(len(trainloader))
epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 30 == 29:
            loss_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0
    lr_sche.step()

print('학습 완료')