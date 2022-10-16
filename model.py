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
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append')

