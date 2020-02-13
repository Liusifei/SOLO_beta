import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms_with_mask
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, cls_color, ins_pred_color
import numpy as np
INF = 1e8
import time
import pdb
from PIL import Image
import os

CLASSES = ('back', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
		   'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
		   'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
		   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
		   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
		   'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
		   'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
		   'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
		   'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
		   'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
		   'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
		   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
		   'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')



@HEADS.register_module
class SoloHead(nn.Module):

	def __init__(self,
				 num_classes,
				 in_channels,
				 feat_channels=256,
				 stacked_convs=4,
				 strides=None,
				 regress_ranges=None,
				 dict_weight = 1.0,
				 loss_cls=dict(
					 type='FocalLoss',
					 use_sigmoid=True,
					 gamma=2.0,
					 alpha=0.25,
					 loss_weight=1.0),
				 conv_cfg=None,
				 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
				 out_path=None):
		super(SoloHead, self).__init__()
		self.num_classes = num_classes
		if loss_cls["use_sigmoid"]:
			self.cls_out_channels = num_classes - 1
		else:
			self.cls_out_channels = num_classes
		self.use_sigmoid = loss_cls["use_sigmoid"]
		self.in_channels = in_channels
		self.feat_channels = feat_channels
		self.stacked_convs = stacked_convs
		self.strides = strides
		self.regress_ranges = regress_ranges
		self.loss_cls = build_loss(loss_cls)
		self.dict_weight = dict_weight
		#self.loss_mask = build_loss(loss_mask)
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.fp16_enabled = False
		self.grid_num=[40,36,24,16,12]
		self.out_path = out_path
		self._init_layers()


	def _init_layers(self):
		self.cls_convs = nn.ModuleList()
		self.mask_convs = nn.ModuleList()
		for i in range(self.stacked_convs):
			chn = self.in_channels if i == 0 else self.feat_channels
			self.cls_convs.append(
				ConvModule(
					chn,
					self.feat_channels,
					3,
					stride=1,
					padding=1,
					conv_cfg=self.conv_cfg,
					norm_cfg=self.norm_cfg,
					bias=self.norm_cfg is None))
			if i == 0:
				self.mask_convs.append(
					ConvModule(
						chn+2,
						self.feat_channels,
						3,
						stride=1,
						padding=1,
						conv_cfg=self.conv_cfg,
						norm_cfg=self.norm_cfg,
						bias=self.norm_cfg is None
						))
			else:
				self.mask_convs.append(
					ConvModule(
						chn,
						self.feat_channels,
						3,
						stride=1,
						padding=1,
						conv_cfg=self.conv_cfg,
						norm_cfg=self.norm_cfg,
						bias=self.norm_cfg is None))
		self.solo_cls = nn.ModuleList([nn.Conv2d(
			self.feat_channels, self.cls_out_channels, 1, stride=1, padding=0) for _ in self.grid_num])
		# self.solo_mask = nn.ModuleList([nn.Conv2d(
			# self.feat_channels, num**2, 1, stride=1, padding=0) for num in self.grid_num])
		self.solo_mask = nn.ModuleList(
			[nn.Conv2d(self.feat_channels, self.grid_num[0]**2, 1, stride=2, padding=0),
			nn.Conv2d(self.feat_channels, self.grid_num[1]**2, 1, stride=1, padding=0),
			nn.Conv2d(self.feat_channels, self.grid_num[2]**2, 1, stride=1, padding=0),
			nn.Conv2d(self.feat_channels, self.grid_num[3]**2, 1, stride=1, padding=0),
			nn.Conv2d(self.feat_channels, self.grid_num[4]**2, 1, stride=1, padding=0)
			])

		
		self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

	def init_weights(self):
		for m in self.cls_convs:
			normal_init(m.conv, std=0.01)
		for m in self.mask_convs:
			normal_init(m.conv, std=0.01)
	   
		bias_cls = bias_init_with_prob(0.01)
		for m in self.solo_cls:
			normal_init(m, std=0.01, bias=bias_cls)
		for m in self.solo_mask:
			normal_init(m, std=0.01)


	def forward(self, feats):
		cls_score, mask_score = multi_apply(self.forward_single, feats, self.solo_cls, self.solo_mask, self.grid_num)
		return cls_score, mask_score

	def forward_single(self, x, solo_cls, solo_mask, grid_num):

		cls_feat = F.upsample_bilinear(x,(grid_num,grid_num))
		mask_feat = x
		for cls_layer in self.cls_convs:
			cls_feat = cls_layer(cls_feat)
		
		cls_score = solo_cls(cls_feat)
		n, _, h, w = mask_feat.shape

		grid_x = torch.arange(w).view(1,-1).float().repeat(h,1).cuda() / (w-1) * 2 - 1
		grid_y = torch.arange(h).view(-1,1).float().repeat(1,w).cuda() / (h-1) * 2 - 1

		x_map = grid_x.view(1, 1, h, w).repeat(n, 1, 1, 1)
		y_map = grid_y.view(1, 1, h, w).repeat(n, 1, 1, 1)
		mask_feat_xy = torch.cat((mask_feat, x_map, y_map), dim=1)

		for mask_layer in self.mask_convs:
			mask_feat_xy = mask_layer(mask_feat_xy)
		mask_score = solo_mask(mask_feat_xy)

		return cls_score, mask_score

	def dice_loss(self,input, target):
		smooth = 1.
		iflat = input.contiguous().view(-1)
		tflat = target.contiguous().view(-1)
		intersection = (iflat * tflat).sum()

		return 1 - ((2. * intersection + smooth) /
			  ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))

	def dict_loss_batch(self, input, target):
		smooth = 1.
		b = input.size(0)
		iflat = input.contiguous().view(b,-1)
		tflat = target.contiguous().view(b,-1)
		intersection = torch.sum(iflat * tflat, dim=1)

		nu = 2. * intersection + smooth
		de = torch.sum(iflat*iflat, dim=1) + torch.sum(tflat*tflat, dim=1) + smooth

		return torch.mean(1-nu/de)

	def loss(self,
				cls_scores,
				mask_preds,
				gt_bboxes,
				gt_labels,
				gt_masks,
				category_targets,
				point_ins,
				img_metas,
				cfg,
				gt_bboxes_ignore=None):
		assert len(cls_scores) == len(mask_preds)
		_, _, b_h, b_w = mask_preds[0].shape

		for i in range(len(self.grid_num)):
			mask_preds[i] = F.sigmoid(F.upsample_bilinear(mask_preds[i], (b_h, b_w)))
		mask_preds = torch.cat([mask_pred for mask_pred in mask_preds], dim=1)

		bound = [self.grid_num[i] ** 2 for i in range(5)]
		bound = [0] + bound
		for i in range(1, 6):
			bound[i] += bound[i - 1]

		num_imgs = len(category_targets)
		for i in range(num_imgs):
			_, i_h, i_w = gt_masks[i].shape
			gt_masks[i] = nn.ConstantPad2d((0, b_w * self.strides[0] - i_w, 0, b_h * self.strides[0] - i_h), 0)(torch.tensor(gt_masks[i]))
			gt_masks[i] = F.upsample_nearest(gt_masks[i].float().unsqueeze(0), (b_h, b_w))[0]

		flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores]
		# need to check images first or dimensions first: image first and then row first
		flatten_cls_scores = torch.cat([cls_score for cls_score in flatten_cls_scores], dim=1).reshape(-1, self.cls_out_channels)

		# calculate loss
		loss_mask = torch.zeros(num_imgs).to(mask_preds[0].device)

		for i in range(num_imgs):
			ind = torch.nonzero(category_targets[i]).squeeze(-1)
			ins_ind = point_ins[i][ind]
			ins_mask = gt_masks[i][ins_ind].to(mask_preds.device)
			pred_mask = mask_preds[i][ind]
			loss_mask[i] = self.dict_loss_batch(pred_mask, ins_mask)

		loss_mask = self.dict_weight * torch.mean(loss_mask)
		category_targets = torch.cat(category_targets)
		num_pos = (category_targets > 0).sum()

		loss_cls = self.loss_cls(flatten_cls_scores, category_targets, avg_factor=num_pos + num_imgs)

		return dict(
			loss_cls=loss_cls,
			loss_mask=loss_mask)


	def loss_ml(self,
				cls_scores,
				mask_preds,
				gt_bboxes,
				gt_labels,
				gt_masks,
				category_targets,
				point_ins,
				img_metas,
				cfg,
				gt_bboxes_ignore=None):
		assert len(cls_scores) == len(mask_preds)
		_, _, b_h, b_w = mask_preds[0].shape
		_, _, b_h3, b_w3 = mask_preds[3].shape

		mask_preds[4] = F.upsample_bilinear(mask_preds[4], (b_h3, b_w3))

		# pdb.set_trace()

		bound = [self.grid_num[i] ** 2 for i in range(5)]
		bound = [0] + bound
		for i in range(1, 6):
			bound[i] += bound[i - 1]

		num_imgs = len(category_targets)
		for i in range(num_imgs):
			_, i_h, i_w = gt_masks[i].shape
			gt_masks[i] = nn.ConstantPad2d((0, b_w * self.strides[0] - i_w, 0, b_h * self.strides[0] - i_h), 0)(torch.tensor(gt_masks[i]))


		flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score
							  in cls_scores]
		# need to check images first or dimensions first: image first and then row first
		flatten_cls_scores = torch.cat([cls_score for cls_score in flatten_cls_scores], dim=1).reshape(-1, self.cls_out_channels)

		# calculate loss
		loss_mask = torch.zeros(len(self.grid_num), num_imgs).to(mask_preds[0].device)
		
		mask_profs = []
		for j in range(len(self.grid_num)):
			mask_profs.append(F.sigmoid(mask_preds[j]))

		for i in range(num_imgs):
			for j in range(len(self.grid_num)):
				mask_ = mask_profs[j][i]
				ind = torch.nonzero(category_targets[i][bound[j]:bound[j + 1]]).squeeze(-1)
				if len(ind) > 0:
					ins_ind = point_ins[i][bound[j]:bound[j + 1]][ind]
					_, b_h_i, b_w_i = mask_.shape
					gt_masks_ = F.upsample_nearest(gt_masks[i].float().unsqueeze(0), (b_h_i, b_w_i))[0]
					ins_mask = gt_masks_[ins_ind].to(mask_.device)		
					pred_mask = mask_[ind]
					loss_mask[j,i] = self.dict_loss_batch(pred_mask, ins_mask)

		loss_mask = self.dict_weight * torch.mean(loss_mask)
		category_targets = torch.cat(category_targets)
		num_pos = (category_targets > 0).sum()

		loss_cls = self.loss_cls(flatten_cls_scores, category_targets, avg_factor=num_pos + num_imgs)

		return dict(
			loss_cls=loss_cls,
			loss_mask=loss_mask)



	def infer_vis(self, img_metas, cls_scores, det_masks, det_bboxes, det_labels, det_scs, out_path="vis_tmp"):

		# ==================vis====================
		# cls_scores_ori
		# ==================vis====================
		ori_h, ori_w, _ = img_metas[0]['ori_shape']
		imgname = img_metas[0]['filename']
		imgname2 = imgname.split('/')
		imgname2 = imgname2[-1][:-4]
		img = np.array(Image.open(imgname))
		det_masks_int = det_masks * 1
		out_ = os.path.join(out_path, imgname2)
		out_cls = os.path.join(out_path, imgname2+'_cls')

		if not os.path.exists(out_):
			os.makedirs(out_)
		if not os.path.exists(out_cls):
			os.makedirs(out_cls)
		
		# # vis cls
		cls_scores_ori = []
		 
		for i in range(5):
			cls_sc = cls_scores[i].clone().detach()
			cls_sc = F.upsample_bilinear(cls_sc, (ori_h, ori_w))
			cls_scores_ori.append(cls_sc)
		cls_all = cls_color(cls_scores_ori, self.use_sigmoid, img)
		 
		for i, cls_ in enumerate(cls_all):
			out_name = os.path.join(out_cls, "scale_%d"%(i) + '.jpg')
			clsim = Image.fromarray(np.uint8(cls_))
			clsim.save(out_name)
		 
		for i in range(5):
			if self.use_sigmoid:
				cls_scores_ori_i = cls_scores_ori[i].sigmoid().cpu()
			else:
				cls_scores_ori_i = cls_scores_ori[i].softmax(dim=1)[:,1:].cpu()

			for j in range(80):
				cls_scores_ori_ij = Image.fromarray(np.uint8(cls_scores_ori_i[0,j,:,:]*255))
				out_name = os.path.join(out_cls, "%02d"%(i) +'_' + "%02d"%(j)+'_'+CLASSES[j]+'.jpg')
				cls_scores_ori_ij.save(out_name)
		
		ins_all, scs_sort = ins_pred_color(img, det_masks_int,
							det_bboxes[:,-1], det_labels, det_scs)
		 
		for i in range(len(ins_all)):
			img2 = Image.fromarray(ins_all[i])
			out_name = os.path.join(out_, "%03d"%(i) + '_' + 'scale'+str(scs_sort[i]) + '.jpg')
			img2.save(out_name)

		# for i in range(det_masks_int.shape[0]):
		#  mas = det_masks_int[i]
		#  lab = det_labels[i]
		#  cls_ = CLASSES[lab]
		#  img_clone = img.copy()
		#  img_clone[:,:,0][mas==1] = 255
		#  img2 = Image.fromarray(img_clone)
		#
		#  out_name = os.path.join(out_, '{}_'.format(cls_) + str(i) + '.jpg')
		#  img2.save(out_name)
		# # ==================vis====================


	@force_fp32(apply_to=('cls_scores', 'mask_preds'))
	def get_bboxes(self, cls_scores, mask_preds, img_metas, cfg, rescale=None, score_thr=True):
		if cfg.get('test_sigmoid', -1):
			return self.get_bboxes_CE(cls_scores, mask_preds, img_metas, cfg, rescale, score_thr)
		else:
			return self.get_bboxes_logi(cls_scores, mask_preds, img_metas, cfg, rescale, score_thr)


   
	def get_bboxes_logi(self,cls_scores, mask_preds, img_metas, cfg, rescale=None, score_thr=True):
		_, _, b_h, b_w = mask_preds[0].shape
		crop_h, crop_w, _ = img_metas[0]['img_shape']
		ori_h, ori_w, _ = img_metas[0]['ori_shape']
		cls_scores_ori = []
		for i in range(5):
			cls_sc = cls_scores[i].clone().detach()
			cls_sc = F.upsample_bilinear(cls_sc, (ori_h, ori_w))
			cls_scores_ori.append(cls_sc)
		# cat dim: default 0
		if self.use_sigmoid:
			flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).sigmoid()
		else:
			flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).softmax(dim=1)
			flatten_cls_scores = flatten_cls_scores[:,1:]  
		for i in range(5):
			mask_preds[i] = F.upsample_bilinear(mask_preds[i],(b_h,b_w))
		mask_preds = torch.cat(mask_preds,dim=1)[0]
		# logit and class (per pixel)
		scores, labels = torch.max(flatten_cls_scores,dim=-1)
		nms_pre = cfg.get('nms_pre', -1)
		if score_thr:
			score_thr = cfg.get('score_thr', -1)
		else:
			score_thr = 0
		mask_thr = cfg.get('mask_thr_binary',-1)
		det_masks = np.array([])
		det_bboxes = np.zeros((0,5))
		det_labels = np.zeros(0).astype(int)
		# store feature scale
		sc_ind = []
		for i in range(5):
			sc_ind.append(torch.zeros((self.grid_num[i] ** 2, 1)) + i)
		sc_ind = torch.cat(sc_ind, axis=0)
		if nms_pre > 0 and len(scores) >=1 and scores.max() > score_thr:
			# HERE: need to change scores for softmax
			valid_inds = torch.nonzero(scores >= score_thr).squeeze()
			if valid_inds.dim()!=0:
				scores = scores[valid_inds]
				labels = labels[valid_inds]
				mask_preds = mask_preds[valid_inds]
				sc_ind = sc_ind[valid_inds]

				if scores.shape[0] < nms_pre:
					nms_pre = scores.shape[0]

				_, topk_inds = scores.topk(nms_pre)
				scores = scores[topk_inds]
				labels = labels[topk_inds]
				mask_preds = mask_preds[topk_inds]
				sc_ind = sc_ind[topk_inds]
				mask_preds = F.upsample_bilinear(mask_preds.unsqueeze(0), (b_h*self.strides[0], b_w*self.strides[0]))
				mask_preds = mask_preds[:, :, :crop_h, :crop_w]
				mask_preds = F.sigmoid(F.upsample_bilinear(mask_preds, (ori_h, ori_w)))[0]
				mask_preds = mask_preds > mask_thr

				masks = self.nms(scores, labels, mask_preds, sc_ind, cfg.nms.iou_thr)
				n = len(masks)
				det_masks = []
				det_bboxes = np.zeros((n, 5))
				det_labels = np.zeros(n).astype(int)
				det_scs = np.zeros(n).astype(int)

				for i in range(n):
					det_bboxes[i, -1] = masks[i][-1]
					det_labels[i] = masks[i][-2]
					det_masks.append(masks[i][0])
					det_scs[i] = masks[i][-1]
				det_masks = np.array(det_masks)

				if self.out_path is not None:
					self.infer_vis(img_metas, cls_scores, det_masks, det_bboxes, det_labels, det_scs, out_path=self.out_path)

		return det_bboxes, det_labels, det_masks


	@force_fp32(apply_to=('cls_scores', 'mask_preds'))
	def get_bboxes_CE(self,cls_scores, mask_preds, img_metas, cfg, rescale=None, score_thr=True):
		_, _, b_h, b_w = mask_preds[0].shape
		crop_h, crop_w, _ = img_metas[0]['img_shape']
		ori_h, ori_w, _ = img_metas[0]['ori_shape']

		cls_scores_ori = []
		for i in range(5):
			cls_sc = cls_scores[i].clone().detach()
			cls_sc = F.upsample_bilinear(cls_sc, (ori_h, ori_w))
			cls_scores_ori.append(cls_sc)

		# cat dim: default 0
		flatten_cls_scores = torch.cat([cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]).softmax(dim=1)		

		for i in range(5):
			mask_preds[i] = F.upsample_bilinear(mask_preds[i],(b_h,b_w))
		mask_preds = torch.cat(mask_preds,dim=1)[0]
		# labels: predicted classes
		scores, labels = torch.max(flatten_cls_scores,dim=-1)

		nms_pre = cfg.get('nms_pre', -1)
		if score_thr:
			score_thr = cfg.get('score_thr', -1)
		else:
			score_thr = 0
		mask_thr = cfg.get('mask_thr_binary',-1)
		det_masks = np.array([])
		det_bboxes = np.zeros((0,5))
		det_labels = np.zeros(0).astype(int)

		# store feature scale
		sc_ind = []
		for i in range(5):
			sc_ind.append(torch.zeros((self.grid_num[i] ** 2, 1)) + i)
		sc_ind = torch.cat(sc_ind, axis=0)
		if nms_pre > 0 and len(scores) >=1 and scores.max() > score_thr:
			# valid_inds: FG indexes
			valid_inds = torch.nonzero(labels!=0).squeeze()
			if valid_inds.dim()!=0:
				scores = scores[valid_inds]
				labels = labels[valid_inds]
				mask_preds = mask_preds[valid_inds]
				sc_ind = sc_ind[valid_inds]
				_, topk = torch.sort(scores, descending=True)

				if len(topk)>0:
					scores = scores[topk]
					labels = labels[topk]
					mask_preds = mask_preds[topk]
					sc_ind = sc_ind[topk]
					labels -= 1
					mask_preds = F.upsample_bilinear(mask_preds.unsqueeze(0), (b_h*self.strides[0], b_w*self.strides[0]))
					mask_preds = mask_preds[:, :, :crop_h, :crop_w]
					mask_preds = F.sigmoid(F.upsample_bilinear(mask_preds, (ori_h, ori_w)))[0]
					mask_preds = mask_preds > mask_thr
					masks = self.nms(scores, labels, mask_preds, sc_ind, cfg.nms.iou_thr)
					# pdb.set_trace()

					n = len(masks)
					det_masks = []
					det_bboxes = np.zeros((n, 5))
					det_labels = np.zeros(n).astype(int)
					det_scs = np.zeros(n).astype(int)
					for i in range(n):
						det_bboxes[i, -1] = masks[i][-1]
						det_labels[i] = masks[i][-2]
						det_masks.append(masks[i][0])
						det_scs[i] = masks[i][-1]
					det_masks = np.array(det_masks)

					if self.out_path is not None:
						self.infer_vis(img_metas, cls_scores, det_masks, det_bboxes, det_labels, det_scs, out_path=self.out_path)

		return det_bboxes, det_labels, det_masks

	def iou_calc(self,mask1,mask2):
		overlap = mask1 & mask2
		union = mask1 | mask2
		iou = float(overlap.sum()+1)/float(union.sum()+1)
		return iou

	def nms(self, scores, labels, masks, sc_ind, iou_threshold=0.5):
		"""
		nms function
		:param boxes: list of box
		:param iou_threshold:
		:return:
		"""
		return_mask = []
		n = len(labels)
		if n > 0:
			masks_dict = {}
			for i in range(n):
				if labels[i].item() in masks_dict:
					masks_dict[labels[i].item()].append([masks[i],labels[i],scores[i],sc_ind[i]])
				else:
					masks_dict[labels[i].item()] = [[masks[i],labels[i],scores[i],sc_ind[i]]]
			for masks in masks_dict.values():
				if len(masks) == 1:
					return_mask.append(masks[0])
				else:
					while (len(masks)):
						best_mask = masks.pop(0)
						if best_mask[0].sum() < 100:
							continue
						return_mask.append(best_mask)
						j = 0
						for i in range(len(masks)):
							i -= j
							if self.iou_calc(best_mask[0], masks[i][0]) > iou_threshold:
								masks.pop(i)
								j += 1
		return return_mask