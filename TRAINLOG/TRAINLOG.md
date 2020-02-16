## 02/13/2020
# Exps with new settings

 1. Restore FPN to default
 2. Conv unit 
 3. [ ] **Multi-scale loss** 
    `root@1034298:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_mlaug.py 8 --work_dir r50_p2_p6_match_ml_3X/`


 4. [ ] **Same-scale loss**
`root@1037871:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_aug.py 8 --work_dir r50_p2_p6_match_3X/`

```
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.265  
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.481  
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.257  
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061  
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.287  
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.235  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.340  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.353  
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109  
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398  
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
```
* worse than r50_p2_p6
* cls loss is actually lower than r50_p2_p6

 - [ ] **Class branch w CE**
 `root@1038535:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_CE.py 8 --work_dir r50_p2_p6_match_CE_3X`

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjg5MDYwNzkxLDkzNzQ2MzkxMiw0NjY1MD
YxOTAsLTc3MDg1MjQ2NiwtMTA4ODIyMjc4MSwtMTU1MzI3Njk5
NF19
-->