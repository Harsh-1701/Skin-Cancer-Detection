# Paper 2 - Current Status

## Research question
Does combining predictive uncertainty and disagreement between EfficientNet-B0 and EfficientNet-B4 improve detection of incorrect predictions?

## Test results
- Test samples: 5,452
- B0 errors: 707 (12.97%)
- B4 errors: 1,119 (20.52%)
- B0 validation-selected alpha: 0.78
- B4 validation-selected alpha: 0.57

## Main findings
- B0 combined AUROC: 0.866; confidence-only AUROC: 0.847.
- B4 combined AUROC: 0.910; confidence-only AUROC: 0.819.
- B0 combined AUPRC: 0.477; confidence-only AUPRC: 0.429.
- B4 combined AUPRC: 0.748; confidence-only AUPRC: 0.511.
- At a 20% review budget, combined error precision was 43.8% for B0 and 75.6% for B4.
- Corrected risk-coverage AURC: B0 combined 0.0297; B4 combined 0.0487. Lower is better.

## Outputs
- Test predictions: test_predictions_model_specific_preprocessing.csv
- Metrics JSON: error_detection_metrics_model_specific_preprocessing.json
- Final comparison table: paper2_final_comparison.csv
- Methods draft: methods_draft.md
- Figures: figure_b0_roc_comparison.png, figure_b4_roc_comparison.png, figure_b0_pr_comparison.png, figure_b4_pr_comparison.png, figure_b0_risk_coverage.png, figure_b4_risk_coverage.png, figure_top20_error_precision.png

## Limitations
- The split is row-stratified.
- Patient- and lesion-level independence has not been established.
- Results are internal test-split findings and do not establish clinical readiness or independent-patient generalization.

## Next steps
1. Verify the exact dataset split and model training procedures from project scripts.
2. Inspect all figures visually.
3. Finalize Methods and Results.
4. Draft Discussion and Limitations.
