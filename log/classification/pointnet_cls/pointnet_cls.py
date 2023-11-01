import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        # 返回全局特徵
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # 3個全連接層
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k) # 輸出:分類數 k=40類
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1) # 計算對數概率
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # nll_loss 的輸入是一個對數概率向量和一個目標標籤:不會計算對數概率
        # 適合網路最後一層是log_softmax
        # 損失函數 nn.CrossEntropyLoss() 和 nn.nll_loss()相同，唯一不同是 nn.CrossEntropyLoss() 去做softmax
        loss = F.nll_loss(pred, target) # 分類損失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat) # 特徵變換正則化損失 
        # 總的損失函數
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
