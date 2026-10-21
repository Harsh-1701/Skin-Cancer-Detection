# Paper 2 Project Status

## Research question
Does disagreement between EfficientNet-B0 and EfficientNet-B4 improve detection of incorrect skin-lesion classifications beyond confidence alone?

## Current stage
Corrected filename-deduplicated evaluation completed. Metrics, predictions, split manifest, and comparison figures have been regenerated.

## Dataset audit
- Original combined dataset: 36,346 rows.
- Unique filenames before resolving label conflicts: 26,331.
- Conflicting filenames excluded: 197, each labeled `akiec` in HAM10000 and `scc` in ISIC2019.
- Evaluation manifest: 26,134 unique filenames across nine classes.
- Split: 18,293 training, 3,920 validation, 3,921 test.
- Split manifest: `split_manifest_deduplicated.csv`.
- Exact filename duplicates across splits: none.

## Preliminary test results
EfficientNet-B0:
- Errors: 296 / 3,921 (7.55%).
- Confidence-only AUROC / AUPRC: 0.8857 / 0.4222.
- Combined AUROC / AUPRC: 0.8934 / 0.4599.
- Combined top-20% precision / recall: 0.2994 / 0.7939.

EfficientNet-B4:
- Errors: 742 / 3,921 (18.92%).
- Confidence-only AUROC / AUPRC: 0.8174 / 0.4957.
- Disagreement-only AUROC / AUPRC: 0.8995 / 0.7198.
- Combined AUROC / AUPRC: 0.9416 / 0.8115.
- Combined top-20% precision / recall: 0.8013 / 0.8477.

## Important limitations
- Filename-level deduplication does not establish patient- or lesion-level independence.
- Overlap among differently named images has not been ruled out.
- The 197 conflicting labels were excluded from this evaluation; the underlying diagnosis conflicts remain unresolved.
- Findings are preliminary and do not establish clinical utility.
- Results and manuscript claims must be rechecked if the dataset, split, preprocessing, or scoring method changes.

## Key artifacts
- `deduplicated_dataset_manifest.csv`
- `duplicate_label_conflicts.csv`
- `split_manifest_deduplicated.csv`
- `test_predictions_model_specific_preprocessing.csv`
- `error_detection_metrics_model_specific_preprocessing.json`
- `paper2_final_comparison.csv`
- `paper2_test_results_table.csv`
- `methods_draft.md`
- `figure_b0_roc_comparison.png`
- `figure_b0_pr_comparison.png`
- `figure_b0_risk_coverage.png`
- `figure_b4_roc_comparison.png`
- `figure_b4_pr_comparison.png`
- `figure_b4_risk_coverage.png`
- `figure_top20_error_precision.png`

## Next tasks
1. Check the exact score definitions and figure labels against the evaluation code.
2. Investigate potential patient/lesion overlap where metadata permits.
3. Regenerate final comparison tables from the corrected metrics.
4. Draft the Results and Discussion with explicit limitations.
5. Review all claims before treating the manuscript as submission-ready.
