import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

# 規一化點雲，使用 centroid 為中心的座標，球半徑為1
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# square_distance函數用來在ball query過程中確定每一個點距離採樣點的距離。
# 函數輸入是兩組點，N為第一組點Src的個數，M為第二組點dst的個數，C為輸入點的通道數（如果是xyz時C=3）
# 函數返回的是兩組點兩兩之間的歐幾里德距離，即NxM的矩陣。
# 在訓練中數據以Mini-Batch的形式輸入，所以一個Batch數量的維度為B
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# 按照輸入的點雲數據和索引返回索引的點雲數據。
# 例如points為Bx2048×3點雲，idx為［5,666,1000,2000」
# 則返回Batch中第5,666,1000,2000個點組成的Bx4x3的點雲集。
# 如果idx為一個［B,D1,...DN］，則它會按照idx中的維度結構將其提取成［B,D1,..DN,C］。
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# farthest_point_sample函數完成最遠點採樣：
# 從一個輸入點雲中按照所需要的點的個數npoint採樣出足夠多的點，
# 並且點與點之間的距離要足夠遠。
# 返回結果是npoint個採樣點在原始點雲中的索引。
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 初始化一個centroids矩陣，用於存儲npoint個採樣點的索引位置，大小為Bxnpoint
    # 其中B為BatchSize的個數
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance矩陣（BxN）記錄某個batch中所有點到某一個點的距離，初始化的值很大，後面會迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest表示當前最遠的點，也是隨機初始化，範圍為O~N，初始化B個；每個batch都隨機有一個初始最遠點
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices初始化為0~（B-1）的數組
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # 直到採樣點達到npoint，否則進行如下迭代：
    for i in range(npoint):
        # 設當前的採樣點centroids為當前的最遠點farthest
        centroids[:, i] = farthest
        # 取出該中心點centroid的坐標
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 求出所有點到該centroid點的歐式距離，存在dist矩陣中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 建立一個mask，如果dist中的元素小於distance矩陣中保存的距離值，則更新distance中的對應值
        # 隨著迭代的繼續，distance矩陣中的值會慢慢變小，其相當於記錄著某個Batch中每個點距離所有已出現的採樣點的最小距離
        mask = dist < distance
        distance[mask] = dist[mask]
        # 從distance矩陣取出最遠的點為farthest，繼續下一輪迭代
        farthest = torch.max(distance, -1)[1]
    return centroids


# query ball_ point函數用於尋找球形鄰域中的點。
# 輸入中radius為球形鄰域的半徑，nsample萬母13域中要術件的點，# new_xyz為centroids點的數據，xyz為所有的點雲數據
# 輸出為每個樣本的每個球形鄰域的nsample個採樣點集的索引［B,S, nsample］
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists： ［B, S, N］記錄S個中心點（new_xyz）與所有點（xyz）之間的歐幾里德距離
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距離大於radius^2的點，其group_index直接置為N;其餘的保留原來的値
    group_idx[sqrdists > radius ** 2] = N
    # 考慮到有可能前nsample個點中也有被賦值為N的點（即球形區域內不足nsample個點），
    # 這種點需要捨棄，直接用第一個點來代替即可
    # group first： 實際就是把 group_idx 中的第一個點的值複製：為［B,S.K] 的維度，便於後面的替換
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等於N的點
    mask = group_idx == N
    # 將這些點的值替換為第一個點的值
    group_idx[mask] = group_first[mask]
    return group_idx # S個group

# Sampling + Grouping主要用於將整個點否分散成局部的group，
# 對每一個 group 都可以用 PointINet 單獨地提取局部的全局特徵。
# Sampling + Grouping分成了sample_and_group和sample_and_group_all兩個函數，
# 其區別在於S ample and_group_all 直接將所有點作為一個group。
#例如：512=npoints sampled in farthest point sampling
# 0.2= radius search radius in local region
# 32 = nsample. how many points in each local region

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 從原點雲通過最遠點採樣挑出的採樣點作為new_xyz：
    # 先用 farthest_point_sample 函數實現最遠點採樣得到採樣點的索引，
    # 再通過 index_points 將這些點的從原始點中挑出來，作為new_xyz
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx) # 中心點
    # idx:[B, point, nsample],代表npoint個球形區域中每個區域的nsample個採樣點的素引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # grouped_xyz減去採樣點日
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    #如果每個點上有新的特徵的維度，則拼接新的特徵與舊的特徵，否則直接返回舊的特徵
    #注：用於拼接點特徵數據和點坐標數據
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

# sample_and_group_all直接將所有點作為一個group npoint=1
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# PointNetSetAbstraction類實現普通的Set Abstraction：
# 首先通過sample_and_group 的操作形成局部group，
# 然後對局部group中的每一個點做MLP操作，最後進行局部的最大池化，得到局部的全局特徵。
class PointNetSetAbstraction(nn.Module):
    #例如：
    # npoint = 128, radius =0.4, nsample=64, in_channel=128 + 3, mlp=[128,128,256] ,group_all=False
    # 128 = npoint points sampled in farthest point sampling 
    # 0.4 = radius: search radius in local region
    # 64 = nsample: how many points in each local region
    # [128, 128,256] = output size for MLP on each point
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        #形成局部的group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        #以下是pointnet操作
        # 對局部group中的每一個點做MLP操作
        # 利用1×1的2d的卷積相當於把每個group當成一個通道 共npoint個通道
        # 對[C+D,nsample］的維度上做逐像素的卷積，結果相當於對單個C+D維度做1d的卷積
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 最後進行局部的最大池化，得到局部的全局特徵
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# PointNetSetAbstractionMSG 類實現 MSC方法的 Set Abstraction：
# 這裡radius_list輸入的是一個list， 例如［0.1,0.2,0.4]；
# 對於不同的半徑做ball guery，將不同半徑下的點雲特徵保存在new_ Points_list中最後再拼接到一起
class PointNetSetAbstractionMsg(nn.Module):
    # 例如：128,［0.2,0.4,0.8］,［32,64, 128］,320,［［64, 64, 128］,［128, 128, 256］,［128, 128,256］］
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S] 特徵數據
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        # 最遠點採樣
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        # 將不同半徑下的點雲特徵保存在 new_points_list
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # query_ball_point函數用於尋找球形鄰域中的點
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # 按照輸入的點雲數據和索引返回索引的點雲數據
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # 拼接點特徵數據和點坐標數據
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            # 最大池化，獲得局部區域的全局特徵
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # 不同半徑下的點雲特徵的列表
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 拼接不同半徑下的點雲特徵
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

# Feature Propagation的實現主要通過線性差值和MLP完成。
# 當點的個數只有一個的時候，採用repeat直接複製成N個點；
# 當點的個數大於一個的時候，採用線性差值的方式進行上採樣
# 拼接上下採樣對應點的SA層的特徵，再對拼接後的每一個點都做一個MLP。
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

