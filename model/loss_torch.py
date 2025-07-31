import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        class_targets = torch.argmax(targets, dim=1).long()  # (N, H, W)
        ce_loss = F.cross_entropy(inputs, class_targets)

        # 2) Dice Loss
        # -------------
        # get per-channel probabilities
        if inputs.shape[1] == 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)

        # compute intersection and union per-sample, per-channel
        # shape: (N, C)
        intersection = (probs * targets).sum(dim=(2, 3))
        total = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1.0 - dice_score.mean()   # average over N×C

        return ce_loss + dice_loss

        # CE = F.cross_entropy(inputs, targets)

        # inputs = F.sigmoid(inputs)
        
        # intersection = (inputs * targets).sum(dim=(2, 3))
        # dice = (2.*intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)

        # dice_loss = 1 - (dice.sum() / (inputs.size(0) * inputs.size(1)))
        # Dice_CE = CE + dice_loss

        # return Dice_CE
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        CE = F.cross_entropy(inputs, targets)

        # if inputs.size(1) == 1:
        inputs = F.sigmoid(inputs)
        # else:
        #     inputs = F.softmax(inputs, dim=1)
        
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = CE + dice_loss
        
        return Dice_BCE

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input_image, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input_image)

    def __call__(self, input_image, target_is_real):
        target_tensor = self.get_target_tensor(input_image, target_is_real)
        return self.loss(input_image, target_tensor)
    

class InclusionLoss(nn.Module):
    def __init__(self):
        super(InclusionLoss, self).__init__()
    
    def forward(self, prediction, target):
        # pred_binary = F.sigmoid(prediction)
        pred_binary = (prediction > 0.5).float()

        inclusion_loss = 0.0
        
        included_pixels = pred_binary * target
        not_included_pixels = (1 - pred_binary) * target
        num_not_included = torch.sum(not_included_pixels)
        num_included = torch.sum(included_pixels)
        
        if num_included > 0:
            inclusion_loss = num_not_included / num_included
        else:
            inclusion_loss = num_not_included
        
        inclusion_loss = inclusion_loss / pred_binary.size(0)
        
        return inclusion_loss
    

class ExclusionLoss(nn.Module):
    def __init__(self):
        super(ExclusionLoss, self).__init__()
    
    def forward(self, prediction, target):
        pred_binary = (prediction > 0.5).float()

        exclusion_loss = 0.0
        
        not_included_pixels = pred_binary * target
        num_not_included = torch.sum(not_included_pixels)
        
        num_pred_pixels = torch.sum(pred_binary)
        if num_pred_pixels > 0:
            exclusion_loss = num_not_included / num_pred_pixels
        else:
            exclusion_loss = 0
        
        # 배치 평균
        exclusion_loss = exclusion_loss / pred_binary.size(0)
        
        return exclusion_loss
    

class ExclusionLoss2(nn.Module):
    def __init__(self):
        super(ExclusionLoss2, self).__init__()
    
    def forward(self, prediction, target):
        # Binarize predictions
        pred_binary = (prediction > 0.5).float()

        # Calculate exclusion loss
        excluded_pixels = pred_binary * (1 - target)  # Pixels predicted as positive but should be negative
        not_excluded_pixels = (1 - pred_binary) * (1 - target)  # Pixels correctly excluded
        num_excluded = torch.sum(excluded_pixels)
        num_not_excluded = torch.sum(not_excluded_pixels)
        
        if num_not_excluded > 0:
            exclusion_loss = num_excluded / num_not_excluded
        else:
            exclusion_loss = num_excluded
        
        # Normalize by batch size
        exclusion_loss = exclusion_loss / pred_binary.size(0)
        
        return exclusion_loss
    
class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    