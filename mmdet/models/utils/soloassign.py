import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms_with_mask
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, cls_color, ins_pred_color
import numpy as np
import scipy.ndimage as ndimage
INF = 1e8
import time
import pdb
from PIL import Image
import os


class SOLOAssign(nn.Module):
	"""
	put solotrans here
	"""
	def __init__(self,
				 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
				 fpn_size = [40, 36, 24, 16, 12],
				 sigma = 0.1):
		super(SOLOAssign, self).__init__()
		self.scale_ranges = scale_ranges
		self.fpn_size = fpn_size
		self.sigma = sigma

	def forward(self, boxes, labels, masks):

		n = len(boxes)
		_, img_h, img_w = masks[0].shape
		# n, _, img_h, img_w = img.size()
		category_targets_batch, point_ins_batch = [], []

		for bn in range(n):
			boxes_current = boxes[bn]
			label_current = labels[bn]
			mask_current = masks[bn]

			category_targets, point_ins = [], []
			for i in range(5):
				category_targets.append(torch.zeros((self.fpn_size[i], self.fpn_size[i])).type(torch.int64))
				point_ins.append(torch.ones((self.fpn_size[i], self.fpn_size[i])) * -1)
			category_targets_current = category_targets
			point_ins_current = point_ins

			obj_num = boxes_current.shape[0]
			x1, y1, x2, y2 = boxes_current[:, 0], boxes_current[:, 1], boxes_current[:, 2], boxes_current[:, 3]
			hl = (y2 - y1) + 1
			wl = (x2 - x1) + 1

			gt_areas = torch.sqrt(hl * wl)
			masks_center = torch.zeros((mask_current.shape[0],2)).to(boxes[0].device)

			for i in range(mask_current.shape[0]):
				cent_ = torch.nonzero(mask_current[i], as_tuple=True)
				cent_ = torch.mean(torch.stack(cent_).float(), dim=1)

				if len(cent_) > 0:
					masks_center[i,:] = cent_
				else:
					masks_center[i,:] = [0.5 * (y1[i] + y2[i]), 0.5 * (x1[i] + x2[i])]

			x_mean = masks_center[:,1]
			y_mean = masks_center[:,0]

			left_raw = (x_mean - self.sigma * wl).clamp(0,img_w-1)
			right_raw = (x_mean + self.sigma * wl).clamp(0,img_w-1)
			top_raw = (y_mean - self.sigma * hl).clamp(0,img_h-1)
			bottom_raw = (y_mean + self.sigma * hl).clamp(0,img_h-1)
			ins_list = torch.range(0, obj_num-1)

			for i in range(len(self.scale_ranges)):

				# num of instances, scale for each instance
				hit_indices = ((gt_areas >= self.scale_ranges[i][0]) &
							   (gt_areas <= self.scale_ranges[i][1])).nonzero()

				if len(hit_indices) > 0:
					hit_indices = hit_indices[:,0]

					hit_indices_order = torch.sort(-gt_areas[hit_indices])[-1]

					hit_indices = hit_indices[hit_indices_order]

					h, w = img_h / self.fpn_size[i], img_w / self.fpn_size[i]

					pos_category = label_current[hit_indices]
					pos_left = (torch.floor(left_raw[hit_indices] / w)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_right = (torch.floor(right_raw[hit_indices] / w)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_top = (torch.floor(top_raw[hit_indices] / h)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_bottom = (torch.floor(bottom_raw[hit_indices] / h)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_instance = ins_list[hit_indices].tolist()

					for j in range(len(hit_indices)):
						pos_left_ = pos_left[j]
						pos_right_ = pos_right[j]
						pos_top_ = pos_top[j]
						pos_bottom_ = pos_bottom[j]

						row_ = np.array(range(pos_top_, pos_bottom_+1)).reshape(-1, 1)
						row_num = row_.shape[0]

						col_ = np.array(range(pos_left_, pos_right_+1)).reshape(1, -1)
						col_num = col_.shape[1]

						row_grid = np.tile(row_, (1, col_num)).reshape(row_num * col_num).tolist()
						col_grid = np.tile(col_, (row_num, 1)).reshape(row_num * col_num).tolist()
						try:
							category_targets_current[i][row_grid, col_grid] = pos_category[j]
						except:
							print(masks_center)
						point_ins_current[i][row_grid, col_grid] = pos_instance[j]

			category_targets_current = torch.cat((category_targets_current[0].flatten(), category_targets_current[1].flatten(), category_targets_current[2].flatten(), category_targets_current[3].flatten(), category_targets_current[4].flatten()), dim=0).type(torch.int64).to(boxes[0].device)
			# points for one instance larger than 3: need to be fixed
			point_ins_current =  torch.cat((point_ins_current[0].flatten(), point_ins_current[1].flatten(), point_ins_current[2].flatten(),point_ins_current[3].flatten(), point_ins_current[4].flatten()), dim=0).type(torch.int64).to(boxes[0].device)

			category_targets_batch.append(category_targets_current)
			point_ins_batch.append(point_ins_current)

		return category_targets_batch, point_ins_batch


class SOLOAssign_MK(nn.Module):
	"""
	put solotrans here
	use downsampled mask
	"""
	def __init__(self,
				 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
				 fpn_size = [40, 36, 24, 16, 12],
				 sigma = 0.2):
		super(SOLOAssign_MK, self).__init__()
		self.scale_ranges = scale_ranges
		self.fpn_size = fpn_size
		self.sigma = sigma

	def forward(self, boxes, labels, masks):

		n = len(boxes)
		_, img_h, img_w = masks[0].shape
		# n, _, img_h, img_w = img.size()
		category_targets_batch, point_ins_batch = [], []

		for bn in range(n):
			boxes_current = boxes[bn]
			label_current = labels[bn]
			mask_current = masks[bn]

			category_targets, point_ins = [], []
			for i in range(5):
				category_targets.append(torch.zeros((self.fpn_size[i], self.fpn_size[i])).type(torch.int64))
				point_ins.append(torch.ones((self.fpn_size[i], self.fpn_size[i])) * -1)
			category_targets_current = category_targets
			point_ins_current = point_ins

			obj_num = boxes_current.shape[0]
			x1, y1, x2, y2 = boxes_current[:, 0], boxes_current[:, 1], boxes_current[:, 2], boxes_current[:, 3]
			hl = (y2 - y1) + 1
			wl = (x2 - x1) + 1

			gt_areas = torch.sqrt(hl * wl)
			masks_center = torch.zeros((mask_current.shape[0],2)).to(boxes[0].device)

			for i in range(mask_current.shape[0]):
				cent_ = torch.nonzero(mask_current[i], as_tuple=True)
				cent_ = torch.mean(torch.stack(cent_).float(), dim=1)

				if len(cent_) > 0:
					masks_center[i,:] = cent_
				else:
					masks_center[i,:] = [0.5 * (y1[i] + y2[i]), 0.5 * (x1[i] + x2[i])]

			x_mean = masks_center[:,1]
			y_mean = masks_center[:,0]

			left_raw_l = (x_mean - self.sigma * wl).clamp(0,img_w-1)
			right_raw_l = (x_mean + self.sigma * wl).clamp(0,img_w-1)
			top_raw_l = (y_mean - self.sigma * hl).clamp(0,img_h-1)
			bottom_raw_l = (y_mean + self.sigma * hl).clamp(0,img_h-1)

			self.sigma /= 2.0
			left_raw = (x_mean - self.sigma * wl).clamp(0,img_w-1)
			right_raw = (x_mean + self.sigma * wl).clamp(0,img_w-1)
			top_raw = (y_mean - self.sigma * hl).clamp(0,img_h-1)
			bottom_raw = (y_mean + self.sigma * hl).clamp(0,img_h-1)

			ins_list = torch.range(0, obj_num-1)

			for i in range(len(self.scale_ranges)):

				# indice of instances, scale for each instance
				hit_indices = ((gt_areas >= self.scale_ranges[i][0]) &
							   (gt_areas <= self.scale_ranges[i][1])).nonzero()

				if len(hit_indices) > 0:
					hit_indices = hit_indices[:,0]
					hit_indices_order = torch.sort(-gt_areas[hit_indices])[-1]
					hit_indices = hit_indices[hit_indices_order]
					h, w = img_h / self.fpn_size[i], img_w / self.fpn_size[i]

					pos_category = label_current[hit_indices]
					pos_mask = mask_current[hit_indices].float()
					# pdb.set_trace()
					for j in range(len(hit_indices)):
						pos_mask[j][:top_raw_l[j].long()] = 0
						pos_mask[j][bottom_raw_l[j].long():] = 0
						pos_mask[j][:,:left_raw_l[j].long()] = 0
						pos_mask[j][:,right_raw_l[j].long():] = 0
					pos_mask2cat = F.upsample_nearest(pos_mask.unsqueeze(0), (self.fpn_size[i], self.fpn_size[i]))[0].long()

					pos_instance = ins_list[hit_indices].tolist()

					pos_left = (torch.floor(left_raw[hit_indices] / w)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_right = (torch.floor(right_raw[hit_indices] / w)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_top = (torch.floor(top_raw[hit_indices] / h)).clamp(0, self.fpn_size[i] - 1).type(torch.int)
					pos_bottom = (torch.floor(bottom_raw[hit_indices] / h)).clamp(0, self.fpn_size[i] - 1).type(torch.int)

					for j in range(len(hit_indices)):
						mask_vindex = torch.nonzero(pos_mask2cat[j], as_tuple=True)

						pos_left_ = pos_left[j]
						pos_right_ = pos_right[j]
						pos_top_ = pos_top[j]
						pos_bottom_ = pos_bottom[j]

						row_ = np.array(range(pos_top_, pos_bottom_+1)).reshape(-1, 1)
						row_num = row_.shape[0]

						col_ = np.array(range(pos_left_, pos_right_+1)).reshape(1, -1)
						col_num = col_.shape[1]

						row_grid = np.tile(row_, (1, col_num)).reshape(row_num * col_num).tolist()
						col_grid = np.tile(col_, (row_num, 1)).reshape(row_num * col_num).tolist()
						try:
							category_targets_current[i][mask_vindex] =  pos_category[j]
							# in case small object vanishes
							category_targets_current[i][row_grid, col_grid] = pos_category[j]
						except:
							print(masks_center)
						point_ins_current[i][mask_vindex] = pos_instance[j]
						point_ins_current[i][row_grid, col_grid] = pos_instance[j]

			category_targets_current = torch.cat((category_targets_current[0].flatten(), category_targets_current[1].flatten(), category_targets_current[2].flatten(), category_targets_current[3].flatten(), category_targets_current[4].flatten()), dim=0).type(torch.int64).to(boxes[0].device)
			# points for one instance larger than 3: need to be fixed
			point_ins_current =  torch.cat((point_ins_current[0].flatten(), point_ins_current[1].flatten(), point_ins_current[2].flatten(),point_ins_current[3].flatten(), point_ins_current[4].flatten()), dim=0).type(torch.int64).to(boxes[0].device)

			category_targets_batch.append(category_targets_current)
			point_ins_batch.append(point_ins_current)

		return category_targets_batch, point_ins_batch