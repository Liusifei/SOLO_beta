import cv2
import torch
import numpy as np
# from torchvision import transforms as T
# from torchvision.transforms import functional as F
import torch.nn.functional as F

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

PALETTE = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

def compute_colors_for_labels():
    """
    Simple function that adds fixed colors depending on the class
    """
    cls_num = len(CLASSES)
    label = range(cls_num)
    colors = PALETTE[:,None].repeat(1,cls_num) * torch.arange(cls_num)[None,:].repeat(3,1)
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def cls_color(cls_scores_ori, use_sigmoid=True, img=None):
	colors = compute_colors_for_labels()
	level = len(cls_scores_ori)
	cls_all = []
	for ii in range(level):
		if use_sigmoid:
			cls_scores_ori_i = cls_scores_ori[ii].sigmoid().squeeze()
		else:
			# cls_scores_ori_i = cls_scores_ori[ii].softmax(dim=1)[:,1:].squeeze()
			cls_scores_ori_i = cls_scores_ori[ii].softmax(dim=1).squeeze()
		cls_scores_ori_i[cls_scores_ori_i<0.1] = 0
		cls_scores_hard = torch.argmax(cls_scores_ori_i, dim=0).cpu()
		cls_scores_color = colors[:,cls_scores_hard]
		if img is not None:
			cls_scores_color = 0.5*cls_scores_color.transpose([1,2,0]) + 0.5*img
		cls_all.append(cls_scores_color)
	return cls_all

def ins_gt_color(data):

	gt_labels = data['gt_labels'][0][0].numpy()

	gt_boxes = data['gt_bboxes'][0][0].numpy()
	category_targets = np.array(data['category_targets'][0])
	point_ins = np.array(data['point_ins'][0])

	img = data['img'][0][0].transpose(1, 0).transpose(2, 1).numpy()
	img_info = data['img_meta'][0].data[0][0]
	img_name = img_info['filename'].split('/')[-1][:-4]
	img_shape = img_info['img_shape']
	ori_shape = img_info['ori_shape']
	crop_h, crop_w, _ = img_shape
	ori_h, ori_w, _ = ori_shape

	mean = img_info['img_norm_cfg']['mean']
	std = img_info['img_norm_cfg']['std']
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	img = (img * std) + mean
	grid_num = [40, 36, 24, 16, 12]
	bound = [grid_num[i] ** 2 for i in range(5)]
	bound = [0] + bound
	for i in range(1, 6):
		bound[i] += bound[i - 1]

	img_list = []
	cls_list = []

	img = img[:crop_h, :crop_w, :]
	img = cv2.resize(img, (ori_w, ori_h))

	if len(data['gt_masks'][0]) == 0:
		img_list.append(img)
		return img_list, [], img_name

	gt_masks = data['gt_masks'][0] > 0
	gt_masks = gt_masks[:,:, :crop_h, :crop_w]
	gt_masks = F.upsample(gt_masks.type(torch.FloatTensor), size=(ori_h, ori_w), mode='nearest')
	gt_masks = gt_masks[0].numpy() > 0
	gt_boxes[:,0] = gt_boxes[:,0] * ori_w / crop_w
	gt_boxes[:,2] = gt_boxes[:,2] * ori_w / crop_w
	gt_boxes[:,1] = gt_boxes[:,1] * ori_h / crop_h
	gt_boxes[:,3] = gt_boxes[:,3] * ori_h / crop_h

	text_color = (0, 0, 255)
	bbox_color = (0, 255, 0)

	# visualize all masks in one image
	for j in range(5):

		category_targets_j = category_targets[bound[j]:bound[j + 1]]
		category_targets_j_ = category_targets_j.reshape((grid_num[j],grid_num[j],1))
		category_targets_j_ = np.tile(category_targets_j_, (1,1,3))
		category_targets_j_ = cv2.resize(np.uint8(category_targets_j_), (ori_w, ori_h),
										 interpolation = cv2.INTER_NEAREST)

		ind = np.nonzero(category_targets_j)[0]
		ins_ind = point_ins[bound[j]:bound[j + 1]][ind]
		gt_masks_j = gt_masks[ins_ind]

		gt_boxes_j = gt_boxes[ins_ind]
		gt_labels_j = gt_labels[ins_ind]
		img_ = img.copy()
		for index, (bbox, label) in enumerate(zip(gt_boxes_j, gt_labels_j)):
			color_mask = np.random.randint(0, 256, (1,3), dtype=np.uint8)

			# bbox_color = (int(color_mask[0,0]),int(color_mask[0,1]),int(color_mask[0,2]))

			# color_mask = np.array([0, 0, 255])

			img_[gt_masks_j[index]] = img_[gt_masks_j[index]] * 0.5 + color_mask * 0.5

			category_targets_j_[category_targets_j_[:,:,0]==gt_labels_j[index]] = color_mask

			bbox_int = bbox.astype(np.int32)
			left_top = (bbox_int[0], bbox_int[1])
			right_bottom = (bbox_int[2], bbox_int[3])
			cv2.rectangle(img_, left_top, right_bottom, bbox_color, thickness=2)
			cv2.rectangle(category_targets_j_, left_top, right_bottom, bbox_color, thickness=2)

			# label = label - 1
			label_text = CLASSES[
				label] if CLASSES is not None else 'cls {}'.format(label)
			if len(bbox) > 4:
				label_text += '|{:.02f}'.format(bbox[-1])
			label_text = str(label) + ':' + label_text
			# cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
			#             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
			cv2.putText(img_, label_text, (bbox_int[0], bbox_int[1] - 2),
						cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1)
			cv2.putText(category_targets_j_, label_text, (bbox_int[0], bbox_int[1] - 2),
						cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1)
		# img_ = img_[:crop_h, :crop_w, :]
		# img_ = cv2.resize(img_, (ori_w, ori_h))
		img_list.append(img_)
		cls_list.append(category_targets_j_)

	return img_list, cls_list, img_name


def ins_pred_color(img, det_masks, det_scores, det_labels, det_scs):

	# det_labels += 1
	img_list = []
	cls_list = []
	text_color = (255, 0, 0)
	bbox_color = (0, 255, 0)
	# index = np.argsort(-det_scores)
	index = np.arange(len(det_scores))
	scs_sort = []

	if len(img.shape) < 3:
		img = np.expand_dims(img, axis=2)
		img = np.tile(img, (1,1,3))
	# visualize all masks in one image
	for i in range(len(det_masks)):
		img_ = img.copy()
		masks_i = det_masks[index[i]].cpu()
		labels_i = det_labels[index[i]] + 1
		scores_i = det_scores[index[i]]

		scs_sort.append(det_scs[index[i]])

		y, x = np.where(masks_i > 0)
		if len(y) > 0:
			x1 = x.min()
			y1 = y.min()
			x2 = x.max()
			y2 = y.max()
			color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
			img_[masks_i > 0] = img_[masks_i > 0] * 0.5 + color_mask * 0.5

			left_top = (x1, y1)
			right_bottom = (x2, y2)
			cv2.rectangle(img_, left_top, right_bottom, bbox_color, thickness=2)

			# label = label - 1
			label_text = CLASSES[
				labels_i] if CLASSES is not None else 'cls {}'.format(labels_i)
			label_text += '|{:.02f}'.format(scores_i)
			label_text = str(labels_i) + ':' + label_text
			# cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
			#             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
			cv2.putText(img_, label_text, (x1, y1 - 2),
						cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 2)

			img_list.append(img_)
		else:
			img_list.append(img_)

	return img_list,scs_sort