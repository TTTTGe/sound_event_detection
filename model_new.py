import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super(Crnn,self).__init__()
        self.relu = nn.ReLU()
        self.batch_norm_0 = nn.BatchNorm2d(num_freq)
        self.conv_1 = nn.Conv2d(num_freq, 16, (3,3))
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.max_pool_1 = nn.MaxPool2d((2,2))
        self.conv_2 = nn.Conv2d(16, 32, (3,3))
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.max_pool_2 = nn.MaxPool2d((2,2))
        self.conv_3 = nn.Conv2d(32, 64, (3,3))
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.max_pool_3 = nn.MaxPool2d((1,2))
        self.conv_4 = nn.Conv2d(64, 128, (3,3))
        self.batch_norm_4 = nn.BatchNorm2d(128)
        self.max_pool_4 = nn.MaxPool2d((1,2))
        self.conv_5 = nn.Conv2d(128, 128, (3,3))
        self.batch_norm_5 = nn.BatchNorm2d(128)
        self.max_pool_5 = nn.MaxPool2d((1,2))
        self.BiGRU = nn.GRU(128,128,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(256,num_class)


    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        x = self.batch_norm_0(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.max_pool_3(x)
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)
        x = self.max_pool_4(x)
        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)
        x = self.max_pool_5(x)
        #repeat conv_5 again
        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)
        x = self.max_pool_5(x)
        x, _ = self.BiGRU(x)
        last_time_step = x[:,-1,:]
        output = self.fc(last_time_step)
        return output



    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
