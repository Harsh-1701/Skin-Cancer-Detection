# Methods Draft

## Study objective
This study evaluates whether disagreement between EfficientNet-B0 and EfficientNet-B4 helps identify incorrect skin-lesion classification predictions beyond confidence-based error detection.

## Dataset and preprocessing
The original combined dataset contained 36,346 rows. Duplicate image filenames were identified across sources. A total of 197 filenames had conflicting labels (`akiec` in HAM10000 and `scc` in ISIC2019); these filenames were excluded from the evaluation manifest rather than assigning an arbitrary label. The resulting manifest contained 26,134 unique image filenames across nine classes.

EfficientNet-B0 and EfficientNet-B4 were evaluated using model-specific input preprocessing at 224 × 224 and 380 × 380 pixels, respectively.

## Data splitting
The deduplicated manifest was divided into training (70%), validation (15%), and test (15%) subsets using two-stage stratified splitting by diagnosis, with a fixed random seed. The resulting split contained 18,293 training, 3,920 validation, and 3,921 test images. The split assignments were saved in `split_manifest_deduplicated.csv`. Both models were evaluated using the same split assignments.

This procedure prevents the same filename from appearing in multiple splits. Patient-level or lesion-level independence was not established, and overlap between differently named images has not been ruled out.

## Error detection
Incorrect classification was treated as the target event for error detection. Confidence-based, disagreement-based, and combined signals were compared. The combination weight was selected on the validation set using AUROC, with AUPRC used as a tie-breaker. The selected weights were 0.92 for EfficientNet-B0 and 0.55 for EfficientNet-B4. Test results were calculated after validation-based selection.

Performance was evaluated using AUROC, AUPRC, and precision, recall, and F1 among the top 20% of test cases flagged as potentially erroneous. These results describe error detection on this test split and should not be interpreted as evidence of clinical utility.

## Preliminary results
On the 3,921-image test set, EfficientNet-B0 had 296 incorrect predictions (7.55%), while EfficientNet-B4 had 742 (18.92%).

For EfficientNet-B0, the combined signal achieved AUROC 0.8934 and AUPRC 0.4599, compared with AUROC 0.8857 and AUPRC 0.4222 for confidence alone. At the top-20% flagging threshold, combined-signal precision was 29.9% and recall was 79.4%.

For EfficientNet-B4, the combined signal achieved AUROC 0.9416 and AUPRC 0.8115, compared with AUROC 0.8174 and AUPRC 0.4957 for confidence alone. At the top-20% flagging threshold, combined-signal precision was 80.1% and recall was 84.8%. The disagreement-only signal also performed strongly for EfficientNet-B4, with AUROC 0.8995 and AUPRC 0.7198.

These findings are preliminary. Patient-level independence, possible overlap among differently named images, and the provenance and representativeness of all data sources require careful consideration before drawing broader conclusions.
