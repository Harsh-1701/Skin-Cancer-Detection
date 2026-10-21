# Results and Discussion Draft

## 4. Results

### 4.1 Evaluation dataset and classification errors

Following filename-level deduplication and exclusion of 197 filenames with conflicting diagnosis labels, the evaluation manifest contained 26,134 unique image filenames across nine diagnostic classes. The dataset was divided into training (18,293 images), validation (3,920 images), and test (3,921 images) subsets using stratified splitting. Both EfficientNet models were evaluated on the same test images.

EfficientNet-B0 produced 296 incorrect predictions among 3,921 test images, corresponding to an error rate of 7.55%. EfficientNet-B4 produced 742 incorrect predictions, corresponding to an error rate of 18.92%. Error detection performance was assessed using confidence-only, disagreement-only, and combined signals.

### 4.2 Error detection performance

For EfficientNet-B0, confidence-only error detection achieved an AUROC of 0.8857 and an AUPRC of 0.4222. Disagreement-only detection performed less well, with an AUROC of 0.7218 and an AUPRC of 0.1717. Combining confidence and disagreement increased performance to an AUROC of 0.8934 and an AUPRC of 0.4599.

For EfficientNet-B4, confidence-only detection achieved an AUROC of 0.8174 and an AUPRC of 0.4957. Disagreement-only detection substantially improved these values, achieving an AUROC of 0.8995 and an AUPRC of 0.7198. The combined signal performed best, with an AUROC of 0.9416 and an AUPRC of 0.8115.

These results indicate that the usefulness of disagreement differed between the two model-specific error detection tasks. For EfficientNet-B0, combining the signals yielded a modest improvement over confidence alone. For EfficientNet-B4, disagreement alone provided a substantial improvement over confidence-only detection, and the combined signal achieved the strongest overall discrimination.

### 4.3 Detection among the highest-risk predictions

The top 20% of test predictions were flagged as potentially erroneous for each detection method. Because the test set contained 3,921 images, approximately 20% of predictions were flagged in each analysis.

For EfficientNet-B0, confidence-only detection achieved a flagged-error precision of 28.9%, recall of 76.7%, and F1 score of 0.4200. Disagreement-only detection achieved a precision of 22.8%, recall of 60.5%, and F1 score of 0.3312. The combined signal achieved a precision of 29.9%, recall of 79.4%, and F1 score of 0.4348.

For EfficientNet-B4, confidence-only detection achieved a precision of 48.9%, recall of 51.8%, and F1 score of 0.5029. Disagreement-only detection achieved a precision of 79.6%, recall of 84.2%, and F1 score of 0.8186. The combined signal achieved a precision of 80.1%, recall of 84.8%, and F1 score of 0.8238.

Thus, for EfficientNet-B4, the disagreement-based and combined approaches identified a considerably higher proportion of incorrect predictions among flagged cases than confidence alone. For EfficientNet-B0, the combined approach improved precision and recall only modestly over confidence-only detection, while disagreement alone was less effective.

### 4.4 Summary of test performance

| Model | Method | AUROC | AUPRC | Top-20% precision | Top-20% recall | Top-20% F1 |
|---|---|---:|---:|---:|---:|---:|
| B0 | Confidence only | 0.8857 | 0.4222 | 0.2892 | 0.7669 | 0.4200 |
| B0 | Disagreement only | 0.7218 | 0.1717 | 0.2280 | 0.6047 | 0.3312 |
| B0 | Combined | 0.8934 | 0.4599 | 0.2994 | 0.7939 | 0.4348 |
| B4 | Confidence only | 0.8174 | 0.4957 | 0.4892 | 0.5175 | 0.5029 |
| B4 | Disagreement only | 0.8995 | 0.7198 | 0.7962 | 0.8423 | 0.8186 |
| B4 | Combined | 0.9416 | 0.8115 | 0.8013 | 0.8477 | 0.8238 |

## 5. Discussion

### 5.1 Principal findings

This study investigated whether disagreement between EfficientNet-B0 and EfficientNet-B4 could improve the detection of incorrect skin-lesion classification predictions beyond confidence-based signals. The results suggest that the contribution of disagreement was model-dependent.

For EfficientNet-B4, disagreement-only detection outperformed confidence-only detection across AUROC, AUPRC, and top-20% flagging metrics. Combining the signals improved performance further. This pattern suggests that disagreement between the two models contained useful information for identifying errors associated with EfficientNet-B4 that was not adequately captured by its confidence scores alone.

For EfficientNet-B0, the pattern differed. Confidence-only detection performed better than disagreement-only detection, while the combined signal provided a smaller improvement over confidence alone. Consequently, disagreement should not be assumed to provide a uniform benefit across models or error detection tasks.

The difference between the two models is an important result rather than a detail to smooth over. It suggests that the utility of cross-model disagreement may depend on the error characteristics of the target model, the relationship between the models' predictions, and the construction of the combined score. Further analysis is required to determine which factors account for the observed asymmetry.

### 5.2 Implications for error detection

The EfficientNet-B4 results suggest that cross-model disagreement may be useful as a signal for prioritizing predictions for review. At the top-20% threshold, approximately 80% of flagged cases were incorrect under the combined approach in this test set. However, this result is specific to the evaluated dataset, models, preprocessing, split, and scoring procedure. It does not establish that the approach would achieve similar performance in another dataset or clinical setting.

The EfficientNet-B0 results provide an important qualification. Since disagreement alone performed worse than confidence alone, disagreement is not necessarily a better error indicator in every setting. A combined approach may be preferable when it consistently adds information beyond confidence, but that benefit should be demonstrated separately for each target model and validated on independent data.

The combined score weights were selected using the validation set, with AUROC as the primary selection criterion and AUPRC as a tie-breaker. The selected weights were 0.92 for EfficientNet-B0 and 0.55 for EfficientNet-B4. These values are specific to the current validation procedure and should not be interpreted as universally optimal weights.

### 5.3 Limitations

Several limitations constrain the interpretation of these findings.

First, deduplication was performed using image filenames. This removed repeated filenames across data sources and prevented the same filename from appearing in multiple splits. However, patient-level and lesion-level independence were not established. Differently named images may still depict the same patient or lesion, and such overlap has not been ruled out. The reported test performance should therefore be interpreted as filename-deduplicated performance, not confirmed patient-independent generalization.

Second, 197 filenames with conflicting labels between HAM10000 and ISIC2019 were excluded from the evaluation manifest. The underlying discrepancies were not resolved, and excluding these cases may affect the class distribution and representativeness of the resulting dataset.

Third, the evaluation used a single deterministic train-validation-test split. Performance may vary across different splits or external datasets. Confidence intervals and statistical comparisons have not yet been calculated, so the observed differences should not be described as statistically significant.

Fourth, the evaluation concerns detection of incorrect model predictions, not diagnosis accuracy in clinical practice. High error detection performance does not establish clinical benefit, safety, improved patient outcomes, or suitability for autonomous decision-making.

Finally, the two models may differ in architecture, preprocessing, learned representations, and error patterns. The current results do not isolate which of these factors explains why disagreement was more useful for EfficientNet-B4 than for EfficientNet-B0.

### 5.4 Future work

Future work should investigate patient- or lesion-level grouping where suitable identifiers are available, assess overlap among differently named images, and evaluate the method on an independent external dataset. Confidence intervals and appropriate statistical comparisons should be added to quantify uncertainty in the observed differences. Additional analyses should examine performance across diagnostic classes and across different flagging thresholds.

Further experiments could also compare the current combined score with alternative disagreement measures and evaluate whether validation-selected weights remain stable across splits. Such analyses would help determine whether the observed improvement for EfficientNet-B4 is robust or specific to the current evaluation setup.

### 5.5 Conclusion

In this filename-deduplicated evaluation, disagreement between EfficientNet-B0 and EfficientNet-B4 provided substantially different benefits for the two model-specific error detection tasks. For EfficientNet-B4, disagreement improved error detection over confidence alone, and the combined signal achieved the strongest observed performance. For EfficientNet-B0, the combined signal provided a smaller improvement, while disagreement alone performed worse than confidence alone.

These findings support further investigation of disagreement-aware error detection as a model-dependent approach. Conclusions about generalization and clinical relevance must await stronger evidence of patient- or lesion-level independence, uncertainty estimates, and external validation.


### Paired bootstrap comparisons

Paired bootstrap comparisons were performed using 2,000 resamples of the test images to estimate 95% confidence intervals for differences between error-detection methods. For EfficientNet-B0, the combined method improved AUROC over confidence-only detection by 0.0078 (95% CI 0.0019 to 0.0142) and AUPRC by 0.0377 (95% CI 0.0215 to 0.0547). However, the paired confidence intervals for top-20% precision, recall, and F1 included zero, leaving the direction of these differences uncertain.

For EfficientNet-B4, the combined method improved AUROC by 0.1242 (95% CI 0.1115 to 0.1381) and AUPRC by 0.3158 (95% CI 0.2843 to 0.3488) relative to confidence-only detection. The paired confidence intervals for top-20% precision, recall, and F1 also excluded zero and favored the combined method.

These results indicate that the observed benefit of combining confidence and disagreement was model-dependent and substantially larger for EfficientNet-B4. The intervals are based on image-level resampling within a single test split. They do not account for possible patient- or lesion-level clustering and should not be interpreted as evidence of patient-independent generalization.
