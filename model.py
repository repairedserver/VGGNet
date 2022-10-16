import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
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