# 2. Related Work

## 2.1 Confidence and predictive uncertainty

Maximum softmax probability is a widely used indicator of a neural network's confidence in its predicted class. For error detection, one minus this probability can be used as an uncertainty score, with larger values indicating lower confidence and therefore higher estimated error risk. This is a convenient baseline, but its usefulness depends on how well the score separates incorrect predictions from correct ones.

Guo et al. (2017) examined the calibration of modern neural networks and showed that high predictive confidence does not necessarily correspond to the same level of empirical accuracy. Calibration concerns the agreement between predicted probabilities and observed outcomes, whereas error detection concerns the ranking of predictions by their likelihood of being incorrect. The two are related, but neither should be treated as a direct substitute for the other. Accordingly, the present study evaluates confidence-derived uncertainty using error-detection metrics rather than treating confidence as a calibrated probability of error.

Lakshminarayanan et al. (2017) studied deep ensembles as a practical approach to predictive uncertainty estimation. Ensembles use information from multiple models, providing a broader view of predictive behavior than the output of a single model alone. The disagreement signal used here is simpler: it records whether EfficientNet-B0 and EfficientNet-B4 predict different classes for the same image. It is a binary indicator, not a full probabilistic estimate of uncertainty, and its value must be established empirically.

## 2.2 Uncertainty estimation in skin-lesion classification

Uncertainty estimation has been investigated directly in skin-lesion classification. Lee et al. (2021) proposed joint dermatological lesion classification and confidence modeling, exploring how confidence-related information can be learned alongside lesion predictions. Their approach differs from the present evaluation, which compares error-detection scores for two existing EfficientNet models rather than jointly training a confidence-estimation component.

Mazoure et al. (2022) introduced DUNEScan, a web-based tool for skin-cancer detection that reports uncertainty alongside model predictions. Their work demonstrates the relevance of communicating predictive uncertainty in a skin-lesion classification setting. However, uncertainty estimation and error ranking are not identical tasks: an uncertainty measure may describe aspects of a prediction's reliability without necessarily providing the best ranking of incorrect predictions under a particular evaluation metric.

Fayyad et al. (2024) evaluated conformal prediction, Monte Carlo dropout, and evidential deep learning for uncertainty quantification across medical-imaging datasets, including pigmented skin lesions. Their work broadens the range of uncertainty approaches considered in medical image analysis. These techniques differ in their assumptions and outputs, whereas the present study focuses on a direct comparison between confidence-derived uncertainty and a binary disagreement signal from two EfficientNet architectures.

## 2.3 Disagreement and ensemble methods in skin-lesion classification

Dakhli and Barhoumi (2024) examined model complementarity and disagreement in skin-lesion classification and proposed an ensemble method based on Dempster–Shafer evidence fusion. Their work is particularly relevant because it investigates how differences between models can contribute to an ensemble's classification behavior.

The objective here is different. Rather than using disagreement primarily to construct a fused classifier, this study treats disagreement as a score for ranking the errors of each target model. For an image on which the two models predict different classes, the disagreement score is one; otherwise, it is zero. The score is evaluated on its own and in combination with confidence-derived uncertainty for the target model. This allows the contribution of disagreement to be measured without assuming that disagreement always signals an incorrect prediction or that it improves both models equally.

## 2.4 Selective prediction and identification of difficult cases

Selective classification examines systems that can abstain from making predictions on selected examples, trading prediction coverage for reduced error risk. Geifman and El-Yaniv (2017) developed a selective-classification framework for deep neural networks in which risk is assessed as coverage changes. This provides a useful basis for evaluating error-detection scores: if high-risk predictions can be ranked effectively, a reviewer or downstream system can prioritize them while retaining the remaining predictions.

Koole et al. (2026) compared five uncertainty-quantification methods on a heterogeneous skin-lesion dataset: Monte Carlo dropout, DropConnect, Flipout, deep ensembles, and deep uncertainty quantification. Their evaluation considered classification performance, calibration, uncertainty decomposition, and risk-coverage behavior. They reported substantial agreement among the methods in their ranking of difficult samples. This finding is relevant because it shows that uncertainty-based identification of difficult cases has already received direct attention in skin-lesion research.

The present study addresses a more specific comparison. Instead of comparing several uncertainty-quantification methods, it evaluates binary disagreement between EfficientNet-B0 and EfficientNet-B4 against each target model's confidence-derived uncertainty. It also evaluates a validation-weighted combination of the two signals. The study therefore examines whether this simple disagreement indicator adds useful information for ranking model errors, and whether that contribution differs between the two target models.

## 2.5 Research gap and study positioning

Prior research has examined confidence calibration, predictive uncertainty, confidence-aware skin-lesion classification, ensemble disagreement, and selective prediction. Studies have also investigated uncertainty estimation and the identification of difficult samples in skin-lesion datasets. The present study does not claim that disagreement or uncertainty-based error detection is a new idea.

Instead, it investigates the empirical value of a specific signal and comparison: whether binary disagreement between EfficientNet-B0 and EfficientNet-B4 improves the ranking of each target model's incorrect predictions beyond confidence-derived uncertainty, and whether a weighted combination performs better than either signal alone. Confidence-only, disagreement-only, and combined scores are compared using validation-selected weights, a held-out test set, AUROC, AUPRC, fixed-budget precision, recall and F1, and paired bootstrap confidence intervals.

The conclusions are limited to the evaluated models, dataset, and split. Filename-level deduplication removes repeated filenames across the evaluation partitions, but it does not establish patient- or lesion-level independence. The study also does not provide external validation or demonstrate clinical utility. Its contribution is an empirical comparison of error-detection signals under these constraints, not evidence that the method is ready for clinical use.