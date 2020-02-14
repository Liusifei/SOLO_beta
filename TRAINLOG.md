## 02/13/2020

 - [ ] **Multi-scale loss** 
    `root@1034298:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_mlaug.py 8 --work_dir r50_p2_p6_match_ml_3X/`

```
2020-02-14 03:22:53,663 - INFO - Epoch [1][50/7330]	lr: 0.00399, eta: 2 days, 21:25:59, time: 0.947, data_time: 0.116, memory: 7966, loss_cls: 0.9290, loss_mask: 1.7424, loss: 2.6714
2020-02-14 03:23:35,193 - INFO - Epoch [1][100/7330]	lr: 0.00465, eta: 2 days, 17:08:22, time: 0.831, data_time: 0.047, memory: 7966, loss_cls: 0.8130, loss_mask: 1.7465, loss: 2.5595
2020-02-14 03:24:17,275 - INFO - Epoch [1][150/7330]	lr: 0.00532, eta: 2 days, 15:58:29, time: 0.842, data_time: 0.074, memory: 7966, loss_cls: 0.7600, loss_mask: 1.6456, loss: 2.4055
2020-02-14 03:25:00,707 - INFO - Epoch [1][200/7330]	lr: 0.00599, eta: 2 days, 15:52:26, time: 0.868, data_time: 0.042, memory: 7966, loss_cls: 0.7361, loss_mask: 1.6300, loss: 2.3661
2020-02-14 03:25:40,350 - INFO - Epoch [1][250/7330]	lr: 0.00665, eta: 2 days, 14:42:07, time: 0.793, data_time: 0.044, memory: 7966, loss_cls: 0.7171, loss_mask: 1.6372, loss: 2.3544
2020-02-14 03:26:22,049 - INFO - Epoch [1][300/7330]	lr: 0.00732, eta: 2 days, 14:25:03, time: 0.834, data_time: 0.062, memory: 7966, loss_cls: 0.7076, loss_mask: 1.6197, loss: 2.3272
2020-02-14 03:27:02,822 - INFO - Epoch [1][350/7330]	lr: 0.00799, eta: 2 days, 14:01:08, time: 0.816, data_time: 0.043, memory: 7966, loss_cls: 0.7042, loss_mask: 1.6243, loss: 2.3284
2020-02-14 03:27:44,510 - INFO - Epoch [1][400/7330]	lr: 0.00865, eta: 2 days, 13:53:01, time: 0.834, data_time: 0.048, memory: 7966, loss_cls: 0.6545, loss_mask: 1.6992, loss: 2.3537
2020-02-14 03:28:26,155 - INFO - Epoch [1][450/7330]	lr: 0.00932, eta: 2 days, 13:46:14, time: 0.833, data_time: 0.040, memory: 7966, loss_cls: 0.6684, loss_mask: 1.5095, loss: 2.1778
2020-02-14 03:29:05,899 - INFO - Epoch [1][500/7330]	lr: 0.00999, eta: 2 days, 13:23:50, time: 0.795, data_time: 0.037, memory: 7966, loss_cls: 0.6581, loss_mask: 1.5518, loss: 2.2098
```

 - [ ] **Same-scale loss**
`root@1037871:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_aug.py 8 --work_dir r50_p2_p6_match_3X/`

```
2020-02-14 03:34:08,462 - INFO - Epoch [1][50/7330]	lr: 0.00399, eta: 2 days, 13:57:27, time: 0.845, data_time: 0.112, memory: 8746, loss_cls: 0.9153, loss_mask: 2.4286, loss: 3.3440
2020-02-14 03:34:42,630 - INFO - Epoch [1][100/7330]	lr: 0.00465, eta: 2 days, 8:00:34, time: 0.683, data_time: 0.053, memory: 8753, loss_cls: 0.8129, loss_mask: 2.3064, loss: 3.1192
2020-02-14 03:35:16,435 - INFO - Epoch [1][150/7330]	lr: 0.00532, eta: 2 days, 5:50:32, time: 0.676, data_time: 0.043, memory: 8753, loss_cls: 0.7533, loss_mask: 2.3142, loss: 3.0675
2020-02-14 03:35:51,760 - INFO - Epoch [1][200/7330]	lr: 0.00599, eta: 2 days, 5:18:39, time: 0.707, data_time: 0.060, memory: 8753, loss_cls: 0.7387, loss_mask: 2.2578, loss: 2.9965
2020-02-14 03:36:26,868 - INFO - Epoch [1][250/7330]	lr: 0.00665, eta: 2 days, 4:55:35, time: 0.702, data_time: 0.075, memory: 8753, loss_cls: 0.7140, loss_mask: 2.2023, loss: 2.9163
2020-02-14 03:37:01,330 - INFO - Epoch [1][300/7330]	lr: 0.00732, eta: 2 days, 4:30:25, time: 0.689, data_time: 0.064, memory: 8753, loss_cls: 0.7052, loss_mask: 2.1694, loss: 2.8746
2020-02-14 03:37:35,537 - INFO - Epoch [1][350/7330]	lr: 0.00799, eta: 2 days, 4:09:03, time: 0.684, data_time: 0.046, memory: 8753, loss_cls: 0.6967, loss_mask: 2.1411, loss: 2.8377
2020-02-14 03:38:09,900 - INFO - Epoch [1][400/7330]	lr: 0.00865, eta: 2 days, 3:54:37, time: 0.687, data_time: 0.050, memory: 8753, loss_cls: 0.6645, loss_mask: 2.1643, loss: 2.8288
2020-02-14 03:38:43,697 - INFO - Epoch [1][450/7330]	lr: 0.00932, eta: 2 days, 3:37:51, time: 0.676, data_time: 0.043, memory: 8753, loss_cls: 0.6635, loss_mask: 2.0161, loss: 2.6796
2020-02-14 03:39:16,872 - INFO - Epoch [1][500/7330]	lr: 0.00999, eta: 2 days, 3:18:43, time: 0.663, data_time: 0.042, memory: 8753, loss_cls: 0.6697, loss_mask: 1.9991, loss: 2.6688
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NTMyNzY5OTRdfQ==
-->