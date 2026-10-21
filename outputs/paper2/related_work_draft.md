# 2. Related Work

## 2.1 Confidence and predictive uncertainty

Neural-network confidence is frequently used as a practical indicator of prediction reliability. A common error-detection score is one minus the maximum predicted class probability, assigning higher scores to predictions with lower confidence. However, confidence calibration and error-ranking performance are distinct properties. Guo et al. (2017) investigated calibration methods for modern neural networks and showed that predicted probabilities may not correspond closely to empirical correctness frequencies. This motivates evaluating confidence-derived scores empirically rather than assuming that the probability assigned to a class fully describes the risk of an incorrect prediction.

Deep ensembles offer another approach to predictive uncertainty estimation. Lakshminarayanan et al. (2017) demonstrated that ensembles of neural networks can provide useful predictive uncertainty estimates. Such approaches use information from multiple models to characterize predictive variability and uncertainty. The present study examines a simpler signal: whether two independently specified model architectures assign different top-predicted classes to the same image. This binary disagreement indicator is not equivalent to the richer uncertainty estimates available from probabilistic ensembles, but it provides a straightforward signal that can be evaluated alongside confidence-derived uncertainty.

## 2.2 Uncertainty estimation in skin-lesion classification

Several studies have investigated uncertainty estimation and confidence modeling specifically for skin-lesion classification. Lee et al. (2021) proposed a framework that jointly models dermatological lesion classification and confidence, using confidence information associated with learned features. Their experiments on ISIC 2018 and ISIC 2019 illustrate how uncertainty-related information can be incorporated into the classification process. This approach differs from the present study, which evaluates error-detection scores for already trained EfficientNet-B0 and EfficientNet-B4 models rather than jointly learning a confidence-estimation mechanism.

Mazoure et al. (2022) introduced DUNEScan, a web-based tool for skin-cancer detection that provides uncertainty estimates alongside predictions from deep neural networks. Their work establishes a further precedent for examining prediction reliability in skin-cancer classification rather than considering class accuracy alone.

Fayyad et al. (2024) evaluated conformal prediction, Monte Carlo dropout, and evidential deep learning for uncertainty quantification across public medical-imaging datasets, including datasets involving pigmented skin lesions. Their study also examined behavior under distribution shifts. These methods differ in their assumptions and mechanisms, but they share the broader objective of characterizing uncertainty in model predictions.

The present study addresses a related but narrower evaluation question. Rather than developing a new uncertainty-estimation architecture or comparing multiple uncertainty-quantification frameworks, it evaluates whether disagreement between two existing classifiers helps rank incorrect predictions beyond the target model's confidence-derived uncertainty. This distinction defines the scope of the comparison rather than implying that uncertainty-based error detection has not previously been investigated.

## 2.3 Disagreement and ensemble methods in skin-lesion classification

Ensemble methods combine information from multiple models to improve classification performance or exploit complementary prediction patterns. In skin-lesion classification, Dakhli and Barhoumi (2024) proposed an explainable ensemble approach based on Dempster–Shafer evidence fusion and model complementarity. Their work is directly relevant because it investigates disagreement-related information as part of an approach intended to improve skin-lesion classification.

Improving the final class prediction and identifying which predictions are likely to be incorrect are related but distinct objectives. A fusion method may improve classification performance without necessarily producing the best ranking of a particular model's errors. Conversely, a score may help prioritize potentially incorrect predictions without changing the underlying classifier's predicted labels.

In this study, disagreement is defined as a binary indicator of whether EfficientNet-B0 and EfficientNet-B4 predict different classes for an image. The indicator is evaluated separately for each target model, both alone and in combination with that model's confidence-derived uncertainty. This setup makes it possible to assess whether the contribution of disagreement differs between the two target models. It does not assume that disagreement necessarily indicates an error, since two models may disagree while one or both predictions are correct.

## 2.4 Selective prediction and identification of difficult cases

Selective classification considers systems that can abstain from making predictions on selected cases, trading prediction coverage for reduced error risk. Geifman and El-Yaniv (2017) developed a selective-classification framework for deep neural networks in which rejected cases are excluded from automatic prediction and risk is evaluated as coverage changes. This provides methodological context for evaluating error-detection scores: a useful score should rank error-prone predictions so that they can be reviewed or withheld preferentially.

More recent work has examined whether different uncertainty-estimation methods identify similar difficult cases. Koole et al. (2026) compared five uncertainty-quantification methods—Monte Carlo dropout, DropConnect, Flipout, deep ensembles, and deep uncertainty quantification—on a heterogeneous skin-lesion dataset. Their study evaluated classification performance, calibration, uncertainty decomposition, and risk-coverage behavior. The authors reported substantial agreement between methods in their ranking of difficult samples, while differences between methods were more pronounced for calibration and uncertainty decomposition.

This study is closely related in its interest in identifying difficult or error-prone skin-lesion cases. However, the methods and comparisons are not identical. Koole et al. investigated multiple uncertainty-quantification methods within a shared modeling framework, whereas the present study examines a binary disagreement signal between two different EfficientNet architectures and compares it with each target model's confidence-derived uncertainty. The present analysis also evaluates a weighted combination of these signals and reports fixed-budget error-detection metrics. These methodological distinctions provide a basis for comparison without implying that the broader task of uncertainty-based error detection is new.

In the current study, AUROC and AUPRC assess how well each signal ranks incorrect predictions across score thresholds. Precision, recall, and F1 assess the errors captured when the highest-risk 20% of test predictions are flagged. This fixed-budget analysis represents a particular operating point, not a guarantee of a specified error rate. Risk-coverage curves provide an additional descriptive view of how the error rate among retained predictions changes as higher-risk cases are excluded.

## 2.5 Research gap and study positioning

Existing research has investigated neural-network calibration, predictive uncertainty, confidence modeling, ensemble disagreement, and selective prediction. Skin-lesion studies have explored uncertainty estimation through confidence-aware classification, uncertainty-reporting tools, conformal prediction, and ensemble methods. Recent work has also compared multiple uncertainty estimators for identifying difficult skin-lesion cases. The motivation for the present study is therefore not that disagreement or uncertainty-based error detection has been unexplored.

Instead, this study evaluates a specific comparison: whether a simple binary disagreement indicator between EfficientNet-B0 and EfficientNet-B4 provides useful information for ranking incorrect predictions beyond the confidence-derived uncertainty of each target model, and whether its contribution differs between the two models. Confidence-only, disagreement-only, and combined scores are compared using validation-selected combination weights, a held-out test split, threshold-independent ranking metrics, fixed-budget flagging metrics, and paired bootstrap comparisons.

The findings characterize error-detection performance for the evaluated models and dataset. Their interpretation is constrained by the filename-level deduplication procedure, unresolved source-label conflicts, and the absence of established patient- or lesion-level independence. The study does not establish clinical utility, external validity, or generalization to independently collected patient populations.



Fayyad, J., Alijani, S., & Najjaran, H. (2024). Empirical validation of conformal prediction for trustworthy skin lesions classification. *Computer Methods and Programs in Biomedicine, 253*, 108231. https://doi.org/10.1016/j.cmpb.2024.108231

Koole, L., Guo, J., & Valdenegro-Toro, M. (2026). Uncertainty identifies difficult samples across methods: A multi-task study on a heterogeneous skin lesion dataset. *UNSURE 2026, MICCAI 2026 Satellite Events*. https://papers.miccai.org/miccai-2026-sat/UNSURE2026_042.html

Lee, G.-H., Ko, H.-B., & Lee, S.-W. (2021). Joint dermatological lesion classification and confidence modeling with uncertainty estimation. *arXiv preprint arXiv:2107.08770*. https://doi.org/10.48550/arXiv.2107.08770

Mazoure, B., Mazoure, A., Bédard, J., & Makarenkov, V. (2022). DUNEScan: A web server for uncertainty estimation in skin cancer detection with deep neural networks. *Scientific Reports, 12*, Article 179. https://doi.org/10.1038/s41598-021-03889-2