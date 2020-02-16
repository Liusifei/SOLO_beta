## 02/13/2020

 - [ ] **Multi-scale loss** 
    `root@1034298:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_mlaug.py 8 --work_dir r50_p2_p6_match_ml_3X/`


 - [ ] **Same-scale loss**
`root@1037871:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_aug.py 8 --work_dir r50_p2_p6_match_3X/`


 - [ ] **Class branch w CE**
 `root@1038535:/instance_v1/SOLO_OURS# ./tools/dist_train.sh configs/solo/r50_p2_p6_match_CE.py 8 --work_dir r50_p2_p6_match_CE_3X`

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk3NzU1MDkwMyw5Mzc0NjM5MTIsNDY2NT
A2MTkwLC03NzA4NTI0NjYsLTEwODgyMjI3ODEsLTE1NTMyNzY5
OTRdfQ==
-->