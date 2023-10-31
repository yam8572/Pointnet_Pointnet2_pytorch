import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# STN3D:T-Net 3*3 transform
# 類似一個 mini-pointnet
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # torch.nn Conv1d(in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
        # channel=6(X,Y,Z,Normal_X,Normal_Y,Normal_z) channel=3(X,Y,Z) 
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 9=3*3
        self.relu = nn.ReLU()

        # 規一化層
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Symmetric function: max poolong
        x = torch.max(x, 2, keepdim=True)[0]
        # x 參數展平 (拉直)
        x = x.view(-1, 1024)

        # 3 個全連接層 fully connect
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # 展平輸出為9個元素展平

        # 展平的對角矩陣: np.array([1,0,0,0,1,0,0,0,1])
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# STNkd:T-Net 64*64 transform, k默認是64 (和STN3D唯一不同是STN3D的in_channel=3or6; STNkd=k=64)
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Symmetric function: max poolong
        x = torch.max(x, 2, keepdim=True)[0]
        # x 參數展平 (拉直)
        x = x.view(-1, 1024)

        # 3 個全連接層 fully connect
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 展平的對角矩陣 np.eye 對角:1 其他:0
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # affine transformation
        x = x.view(-1, self.k, self.k)
        return x

# PointNet編碼器
class PointNetEncoder(nn.Module):
    # global_feat:是否要做全局特徵 feature_transform:是否需做特徵轉換 in_channel default=3(X,Y,Z)
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        # STN3D:T-Net 3*3 transform
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # 若需做特徵轉換
        if self.feature_transform:
            #  STNkd:T-Net 64*64 transform
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()# batchsize,3(XYZ座標) or 6(XYZ座標+法向量),1024(一個物體所取的點的數目)
        trans = self.stn(x) # STN3D:T-Net 3*3 transform
        x = x.transpose(2, 1) # 轉置 交換一個tensor的兩個維度
        # 若維度超過3
        if D > 3:
            # 下兩行等同 x,feature = x.spilt(3,dim=2)
            feature = x[:, :, 3:] 
            x = x[:, :, :3]
        # 對輸入的點雲進行輸入轉換(input transform)
        # input transform: 計算2個tensor的矩陣乘法
        # bmm 是兩個三維張量相乘，2個輸入tensor 維度是 x= (b x n x m) 和 trans=(b x m x p)
        # 第一維b代表batch_size，輸出為(b x n x p) >> (n x m) (m x p)=(n x p)
        x = torch.bmm(x, trans)# trans = TN3D:T-Net
        if D > 3:
            # 做拼接
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))# MLP 得 n x 64

        # 是否需做 feature transform
        # 是 >> 做 STNkd:T-Net 64*64 transform
        if self.feature_transform:
            trans_feat = self.fstn(x) # 做 STNkd:T-Net 64*64 transform
            x = x.transpose(2, 1)
            # 對輸入的點雲進行特徵轉換(feature transform)
            # feature transform: 計算2個tensor的矩陣乘法
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x # 局部特徵
        x = F.relu(self.bn2(self.conv2(x))) # MLP
        x = self.bn3(self.conv3(x)) # MLP
        x = torch.max(x, 2, keepdim=True)[0] # 最大池化的全局特徵
        x = x.view(-1, 1024) # 展平
        # 需要返回的是否是全局特徵?
        if self.global_feat:
            return x, trans, trans_feat # 返回全局特徵
        else: # 分割
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            # 返回局部特徵與全局特徵的拼接 x=展平的全局特徵 pointfeat=局部特徵
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 對特徵轉換矩陣做正規化
# contrain the feature transformation matrix to be close to orthogonal matrix
# A:T-net 64*64 Lreg = || I-transpose(AA)||2F
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    # I單位矩陣: torch.eye(n,m=None, out=None) 返回一個2維張量，對角線位置全1其他位置0
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    # 正則化損失函數 trans = TN3D:T-Net trans * trans轉置 - I單位矩陣
    # torch.norm 求泛數
    # torch.mean 求平均
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
