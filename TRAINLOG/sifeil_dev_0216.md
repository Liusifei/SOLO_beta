---


---

<h2 id="changelog-0216-sifeil">Changelog 0216 (sifeil)</h2>
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> <a href="http://train.py">train.py</a>
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> save args to work_dir</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> fix distributeparallel for latest mmcv</li>
</ul>
</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Add unsample_bilinear to p2 (1/4-&gt;1/8) and p6 (1/64-&gt;1/32), see solo_head, fowrard()</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Add GN to coord conv</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> FPN remains default</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Add back dict_loss_batch() and fixed loss()</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Change mask_preds and gt_masks to 2<em>b_h and 2</em>b_w- -
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> mask loss done at 1/4 for all the levels</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> for 2x for pos mask only</li>
</ul>
</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> check epoch 14: 20.6/20.0 (pr/rc)</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> check epoch 36:</li>
</ul>
<p><code>2020-02-16 22:19:50,321 - INFO - Start running, host: root@1037871, work_dir: /instance_v1/SOLO_OURS/r50_p2_p6_match_aug_0216</code></p>
<p><code>2020-02-17 06:23:33,323 - INFO - Start running, host: root@1034753, work_dir: /instance_v1/SOLO_OURS/r50_p2_p6_match_1X</code></p>
<p><code>./tools/train.py --local_rank=0 configs/solo/r50_p2_p6_match-v2.py --launcher pytorch --work_dir r50_p2_p6_match_1X_v2</code></p>
<p><code>./tools/train.py --local_rank=6 configs/solo/r50_p2_p6_match-v3.py --launcher pytorch --work_dir r50_p2_p6_match_1X_v3</code></p>
<p><code>./tools/train.py --local_rank=0 configs/solo/r50_p2_p6_match-v4.py --launcher pytorch --work_dir lossx4_match_1Xv4</code> (loss at 1/2)</p>
<p><code>./tools/train.py --local_rank=5 configs/solo/r50_p2_p6_match_augv2.py --launcher pytorch --work_dir caffesgd_match_randscale</code></p>
<h2 id="ablations-considered-bold-is-better">Ablations considered (bold is better)</h2>
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Unsample <strong>before</strong> or after branch-net (mask branch)</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled="">  <code>r50_p2_p6_match-v2.py</code> Focal loss, gamma=2.20, alpha=0.3:  <strong>no difference</strong></li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled="">  tried score_thr=0.05, <strong>no difference</strong></li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> <code>r50_p2_p6_match-v3.py</code> add extra conv, add relu on top of fpn head</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> <code>/instance_v1/SOLO_OURS/caffesgd_match_randscale</code>:
<ul>
<li>tried caffe sgd</li>
<li><code># pred_mask = F.upsample_bilinear(F.sigmoid(mask_preds[i][ind]).unsqueeze(0), (2*b_h, 2*b_w)).squeeze()</code><br>
changed to:<br>
<code>pred_mask = F.sigmoid(F.upsample_bilinear(mask_preds[i][ind].unsqueeze(0), (2*b_h, 2*b_w)).squeeze())</code></li>
<li>no relu after fpn*</li>
</ul>
</li>
</ul>

