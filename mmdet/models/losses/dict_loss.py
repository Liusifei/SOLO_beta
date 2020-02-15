import torch.nn as nn
import torch.nn.functional as F
import pdb
from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module
class DictLoss(nn.Module):
	"""IoU loss for each channel"""
	def __init__(self, smooth = 1.0, loss_weight=1.0):
		super(DictLoss, self).__init__()
		self.smooth = 1.0
		self.loss_weight = loss_weight

	def forward(self, pred, target):
		b = pred.size(0)
		iflat = pred.contiguous().view(b,-1)
		tflat = target.contiguous().view(b,-1)
		intersection = torch.sum(iflat * tflat, dim=1)

		nu = 2. * intersection + smooth
		de = torch.sum(iflat*iflat, dim=1) + torch.sum(tflat*tflat, dim=1) + smooth

		return self.loss_weight * torch.mean(1-nu/de)		
		