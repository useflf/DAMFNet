import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ST_Attention import ChannelsAttentionModule
from model.ST_Attention import TemporalAttentionModule
from model.ST_Attention import ResBlock_CTAM

# CAM(in_channels=[32,16], out_channels=[32,32])
class CAM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CAM, self).__init__()

        self.channel_attention1 = ChannelsAttentionModule(4)
        self.channel_attention2 = ChannelsAttentionModule(12)
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(20, 1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1)),
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=(4, 1),padding='same'),
            # nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=(1, 6),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 3))
        )
        self.convs22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(3, 1), padding=1, stride=2,output_padding=1),
            nn.BatchNorm2d(out_channels[1])
        )

    def forward(self, x1,x2):
        # 32*60*4
        x1 = self.channel_attention1(x1) #32*600*4
        s_map1 = x1
        # print(x1.shape)
        x1 = self.convs1(x1)             # 32*60*4
        x2 = self.channel_attention2(x2) #16*30*12
        s_map2 = x2
        x2 = self.convs2(x2)
        pre_x2 = x2
        x2 = self.convs22(x2) #32*60*4
        out = torch.cat((x1, x2), 1) # 64*60*4
        return out,s_map1,s_map2, pre_x2,x2

# TAM(in_channels=[64,64], out_channels=[64,64])
class TAM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TAM, self).__init__()

        self.spatial_attention1 = TemporalAttentionModule(37)
        self.spatial_attention2 = TemporalAttentionModule(7)
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(8, 1),padding=0),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[0])
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=(1, 6),padding='same'),
            # nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=(4, 1),padding='same'),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 3))
        )
        self.convs22 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=(5, 2), padding=0, stride=(4,1),output_padding=(1,0)),
            nn.BatchNorm2d(out_channels[1])
            #nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

    def forward(self, x1,x2):
        # 30*4*64
        x1 = self.spatial_attention1(x1)    #64*37*4
        t_map1 = x1
        x1 = self.convs1(x1)                #64*30*4
        x2 = self.spatial_attention2(x2)    #64*7*12
        t_map2 = x2
        # print(x2.shape)
        x2 = self.convs2(x2)
        pre_x2 = x2
        x2 = self.convs22(x2)#64*30*4
        # print(x1.shape,x2.shape)
        out = torch.cat((x1, x2), 1)        # 128*30*4
        return out,t_map1,t_map2,pre_x2,x2

class Inception_eeg(nn.Module):
    def __init__(self, in_channels):
        super(Inception_eeg, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)),
                                     nn.Conv2d(16, 24, kernel_size=(3,1), padding='same'),
                                     # nn.Conv2d(24, 24, kernel_size=(3,1), padding='same')
                                     )

        self.branch5 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=(1,1)),
                                     nn.Conv2d(16, 24, kernel_size=(5,1), padding='same')
                                     )

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=(1,1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch5, branch3, branch_pool), dim=1)

#out_channels = [8,16] kernel_size1,kernel_size3,kernel_size5,kernel_size
#(1,1),(3,1)(5,1),(7,1),(1,1)
class Inception_hbr(nn.Module):
    def __init__(self, in_channel,out_channels,kernel_size1,kernel_size3,kernel_size5,kernel_size):
        super(Inception_hbr, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1))

        self.branch3 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1),
                                     nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size3, padding='same'),
                                     # nn.Conv2d(24, 24, kernel_size=(5,1), padding='same')
                                     )

        self.branch5 = nn.Sequential(nn.Conv2d(in_channel, out_channels[0], kernel_size=kernel_size1),
                                     nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size5, padding='same')
                                     )

        self.branch_pool = nn.Conv2d(in_channel, out_channels[1], kernel_size=kernel_size)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat((branch1, branch5, branch3,branch_pool), dim=1)
class MYFusion(nn.Module):
    def __init__(self):
        super(MYFusion, self).__init__()

        self.convs1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 4),padding='same'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 4), padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        # 32*600*4

        self.inception1_1 = Inception_eeg(32)

        self.convs1_3 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=32, kernel_size=(1,1),padding='same'),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )  #32*150*4

        self.convs1_4 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=64, kernel_size=(1,1),padding='same'),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )  #64*37*4

        self.convs2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 12), padding='same'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )  #16*30*24
        self.inception2_1 = Inception_hbr(16,[8,16],(1,1),(1,3),(1,5),(1,1))

        self.convs2_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=16, kernel_size=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )  #16*30*12

        self.convs2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(10, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )  # 32*15*12
        self.inception2_2 = Inception_hbr(32,[16, 32],(1,1),(3,1),(5,1),(1,1))

        self.convs2_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        ) #64*7*12

        self.channel_attention = CAM(in_channels=[32,16], out_channels=[32,32])
        self.temporal_attention = TAM(in_channels=[64,64], out_channels=[64,64])

        self.conv_center =nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), padding='same'),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.res_ctam = ResBlock_CTAM(256,64)


#-------支流eeg------
        self.fc1_0 = nn.Linear(64*37*4, 1024)
        self.fc1_1 = nn.Linear(1024, 2)

 #'''   支流hbr    '''
        self.fc2_0 = nn.Linear(64*7*12, 1024)
        self.fc2_1 = nn.Linear(1024, 2)
#''' fusion '''
        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.Linear(256*30*4, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self,x1,x2):
        """
        x: (batch,h*w*channels)
        eeg: 600*8*1
        """
        x2 = x2.to(torch.float32)
        x1 = x1.view(x1.shape[0], 1, x1.shape[1], 8)
        x2 = x2.view(x2.shape[0], 1, x2.shape[1], 24)
        x1 = self.convs1_1(x1) #   32*600*4
        x2 = self.convs2_1(x2)
        x2 = self.inception2_1(x2)
        x2 = self.convs2_1_1(x2)  # 16*30*12
        ys_x1 = x1
        ys_x2 = x2
        x_c,s_1,s_2,pre_ts,ts = self.channel_attention(x1, x2)  # 64*60*4

        x1 = self.inception1_1(x1)
        x1 = self.convs1_3(x1)
        x1 = self.inception1_1(x1)
        x1 = self.convs1_4(x1)  #   64*37*4

        x2 = self.convs2_2(x2) #   32*15*12
        x2 = self.inception2_2(x2)
        x2 = self.convs2_2_1(x2)  #64*7*12
        yt_x1 = x1
        yt_x2 = x2
        x_t,t_1,t_2,pre_tt,tt = self.temporal_attention(x1, x2)  # 128*30*4
        # fusion:

        x_c = self.conv_center(x_c)          #  128*30*4
        out_c =  torch.cat((x_t, x_c), 1)    #  256*30*4
        # print(out_c.shape) #
     # 融合后的进一步attention
        out_c = self.res_ctam(out_c)     # out_res-stam
     #    out_c = self.res_ctam(x_c)
        #  print('out_c',out_c) #

        fusion_feats = out_c.view(out_c.shape[0],-1)
        x = F.sigmoid(self.fc0(fusion_feats))
        x = self.fc1(self.dropout(x))
        features = self.fc2(x)
        out_fusion =F.softmax(features)

        # eeg: 64*37*2
        x1 = x1.view(x1.shape[0],-1)
        x1 = F.sigmoid(self.fc1_0(x1))
        features1 = self.fc1_1(self.dropout(x1))
        out1 = F.softmax(features1)
        # hbr: 32*15*6
        x2 = x2.view(x2.shape[0],-1)
        x2 = F.sigmoid( self.fc2_0(x2))
        features2 = self.fc2_1(self.dropout(x2))
        out2 = F.softmax(features2)

        return out1,out2, out_fusion, features1,features2,features,ys_x1,ys_x2,s_1,s_2,pre_ts,ts,yt_x1,yt_x2,t_1,t_2,pre_tt,tt


if __name__ == "__main__":
    pass