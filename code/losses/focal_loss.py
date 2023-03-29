import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N,L, C]
        target: [N, L]
        """
        if len(input.shape) ==3 and len(target.shape)==2:
            input= input.reshape(-1, input.shape[-1])
            target= target.reshape(-1)


        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


if __name__=="__main__":
    import torch

    batch_size = 4
    seq_len = 10
    input_dim = 50
    num_classes = 5

    # 构造随机的输入数据
    input_data = torch.rand(batch_size, seq_len, num_classes)
    print(len(input_data.shape))
    input_data=input_data.reshape(-1,num_classes)

    # 构造随机的标签数据
    label_data = torch.randint(num_classes, size=(batch_size, seq_len))
    label_data = label_data.reshape(-1)

    loss_fuc=FocalLoss()
    loss=loss_fuc(input_data,label_data)
    print(loss)