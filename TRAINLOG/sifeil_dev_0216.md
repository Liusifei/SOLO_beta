---


---

<h2 id="changelog-0216-sifeil">Changelog 0216 (Sifeil)</h2>
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
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> check epoch 14:</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled=""> check epoch 36:</li>
</ul>
<p><code>2020-02-16 22:19:50,321 - INFO - Start running, host: root@1037871, work_dir: /instance_v1/SOLO_OURS/r50_p2_p6_match_aug_0216</code></p>
<p><code>2020-02-17 06:23:33,323 - INFO - Start running, host: root@1034753, work_dir: /instance_v1/SOLO_OURS/r50_p2_p6_match_1X</code></p>
<h2 id="ablations-considered-bold-is-better">Ablations considered (bold is better)</h2>
<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled=""> Unsample <strong>before</strong> or after branch-net (mask branch)</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked="true" disabled="">  Focal loss, gamma=1.50, alpha=0.4: <code>r50_p2_p6_match-v2.py</code></li>
</ul>

