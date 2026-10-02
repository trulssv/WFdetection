# evaluation/

Metrics that compare lifted fields with analytic wavefront sets from
`data/phantoms.py`.

## `metrics.py`

* `roc_auc(score, positive, negative)`: rank-based ROC AUC (Mann–Whitney U)
  of a score field for two voxel masks. Threshold-free, so it compares raw
  coefficients, posteriors and network outputs on equal terms.
* `label_masks(membership)` takes positives at ≥ 0.5 and negatives at < 0.01
  of a soft label. Voxels in the gap between them are ignored, so blur at the
  boundary of the label is not counted as an error.
* `t_localization(c, grid, x, theta)` tests the sign and offset conventions of
  the canonical relation directly. For each true WF sample it finds where the
  coefficient line $c[\varphi_k, s_i, :]$ peaks in $t$, compared with the
  predicted $t=x\cdot\omega^\perp(\theta)$:
  * `local_error`: signed error in cells within a ±window;
  * `global_hit_rate`: fraction of samples whose global maximum along $t$ is
    within 2 cells. This can be lowered by other edges tangent to the same
    ray, which is correct behaviour, not an error.

Planned: curve extraction from predicted fields (non-maximum suppression in
$\theta$ plus tracking along $X_1$), for visualization and for comparison with
FBP edge detection.
