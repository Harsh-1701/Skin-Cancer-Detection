# Methods

## 1. Study objective

This study evaluated whether disagreement between EfficientNet-B0 and EfficientNet-B4 improves the detection of incorrect skin-lesion classification predictions beyond confidence-based error detection.

## 2. Dataset and preprocessing

The original combined dataset contained 36,346 rows. Duplicate image filenames occurred across data sources. A total of 197 filenames had conflicting diagnosis labels, recorded as `akiec` in HAM10000 and `scc` in ISIC2019. These filenames were excluded rather than assigning an arbitrary label.

The resulting evaluation manifest contained 26,134 unique image filenames across nine diagnostic classes. The manifest was used to construct the training, validation, and test splits.

Model-specific deterministic evaluation preprocessing was applied. EfficientNet-B0 used an input size of 224 × 224 pixels, while EfficientNet-B4 used 380 × 380 pixels. The evaluation transforms included normalization and tensor conversion. Both models were evaluated on the same split assignments, using their respective input sizes.

## 3. Data splitting

The deduplicated manifest was divided into training, validation, and test subsets using two-stage stratified splitting by diagnosis, with a fixed random seed of 42. The resulting split contained 18,293 training images, 3,920 validation images, and 3,921 test images.

Split assignments were saved in `split_manifest_deduplicated.csv`. Filename-level deduplication prevented identical filenames from appearing in multiple splits. However, patient-level and lesion-level independence were not established, and overlap among differently named images has not been ruled out.

## 4. Error-detection methods

An incorrect classification was defined as an error event. Three error-detection signals were compared for each target model:

1. **Confidence-only:** the model's classification uncertainty, calculated as one minus its predicted confidence.
2. **Disagreement-only:** a binary indicator equal to one when EfficientNet-B0 and EfficientNet-B4 predicted different classes, and zero otherwise.
3. **Combined:** a weighted combination of classification uncertainty and model disagreement.

The combined error score was defined as:

\[
s_{\mathrm{combined}} =
\alpha(1-c) + (1-\alpha)d
\]

where \(c\) is the target model's predicted confidence, \(d\) is the binary disagreement indicator, and \(\alpha\) is the combination weight. Higher scores indicate greater estimated risk of an incorrect prediction.

The combination weight was selected separately for each target model using the validation set. Candidate weights were evaluated on a grid from 0 to 1 in increments of 0.01. AUROC was the primary selection criterion, with AUPRC used to break ties. The selected weights were 0.92 for EfficientNet-B0 and 0.55 for EfficientNet-B4. The test set was not used to select these weights.

## 5. Evaluation metrics

Error-detection performance was assessed using the area under the receiver operating characteristic curve (AUROC), the area under the precision-recall curve (AUPRC), and precision, recall, and F1 score among the highest-risk predictions.

For each method, the top 20% of test cases ranked by error score were flagged as potentially erroneous. The number flagged was defined as the ceiling of 20% of the test-set size. For 3,921 test images, this resulted in 785 flagged cases per method.

Flagged-error precision measured the proportion of flagged cases that were incorrect. Recall measured the proportion of all incorrect predictions captured among the flagged cases. F1 was calculated as the harmonic mean of precision and recall. Error rates among flagged and unflagged cases were also recorded.

## 6. Uncertainty estimation and paired comparisons

Image-level bootstrap resampling was used to estimate 95% percentile confidence intervals for error-detection metrics. The bootstrap procedure requested 2,000 resamples and used random seed 42. Resampling was performed over test rows and therefore did not account for possible clustering of images by patient or lesion.

Paired bootstrap comparisons were conducted to estimate confidence intervals for differences between error-detection methods. The same resampled test rows were used to compare methods within each replicate. The paired comparison procedure used 2,000 bootstrap replicates and random seed 123. Differences were defined as the metric for the first method minus the metric for the comparison method; positive differences favored the first method.

These uncertainty estimates characterize performance on the evaluated test split. They do not establish patient- or lesion-independent generalization or external validity.

## 7. Scope of evaluation

The evaluation focused on identifying incorrect model predictions rather than assessing diagnostic performance in clinical practice. The results do not by themselves establish clinical utility, safety, improved patient outcomes, or suitability for autonomous decision-making. External validation and evaluation with appropriate patient- or lesion-level grouping remain necessary to assess generalizability.
