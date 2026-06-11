# Evaluation Metrics After Applying Class Imbalance Handling Techniques

## 1. Overview

This section documents the evaluation metrics obtained after applying different class imbalance handling techniques across four dataset modalities: image, audio, text, and tabular data. The project, **"Comparative Analysis of Class Imbalance Handling Techniques Across Multi-Modal Datasets: Image, Audio, Text, and Tabular Data"**, evaluates whether balancing strategies improve minority class recognition while preserving overall classification performance.

In imbalanced datasets, the majority class can dominate the learning process, causing high overall accuracy but poor minority class prediction. Therefore, accuracy alone is not sufficient for assessing model performance. Metrics such as Precision, Recall, F1-Score, Macro F1-Score, ROC-AUC, PR-AUC, Specificity, and Balanced Accuracy provide a more reliable interpretation of model behavior under class imbalance.

The metric values reported in this document are realistic placeholder values intended for documentation structure and comparative interpretation. These values must be replaced with actual experimental results after model training, validation, and testing are completed.

## 2. Evaluation Metrics Used

- **Accuracy:** Measures the proportion of correctly classified samples among all samples. In imbalanced datasets, accuracy may be misleading because a model can achieve high accuracy by favoring the majority class.
- **Precision:** Measures the proportion of correctly predicted positive samples among all samples predicted as positive. It is important when false positives are costly.
- **Recall:** Measures the proportion of actual positive samples correctly identified by the model. It is especially important for evaluating minority class detection.
- **F1-Score:** Harmonic mean of Precision and Recall. It provides a balanced measure when both false positives and false negatives are important.
- **Macro F1-Score:** Computes F1-Score independently for each class and averages them equally. This metric is highly relevant for imbalanced datasets because it gives equal importance to minority and majority classes.
- **Weighted F1-Score:** Computes the average F1-Score weighted by class support. It reflects overall performance while considering class distribution.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across different classification thresholds.
- **PR-AUC:** Measures the area under the Precision-Recall curve. It is particularly informative for imbalanced datasets because it focuses on positive class prediction quality.
- **Specificity:** Measures the proportion of actual negative samples correctly identified. It helps evaluate majority or negative class discrimination.
- **Balanced Accuracy:** Average of sensitivity and specificity. It is more reliable than standard accuracy when class distributions are unequal.
- **Confusion Matrix:** Summarizes true positives, true negatives, false positives, and false negatives. It directly shows how balancing affects minority class prediction errors.

## 3. Image Dataset Results

The image dataset evaluation focuses on whether balancing techniques improve recognition of minority visual classes without significantly increasing false positives. In image classification, data augmentation and loss-based reweighting are often effective because they increase visual diversity and reduce majority class dominance during training.

| Dataset / Modality | Balancing Technique | Accuracy | Precision | Recall | F1-Score | Macro F1-Score | Weighted F1-Score | ROC-AUC | PR-AUC | Specificity | Balanced Accuracy | Confusion Matrix Summary | Key Observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Image | Baseline without balancing | 0.88 | 0.81 | 0.68 | 0.74 | 0.71 | 0.86 | 0.84 | 0.76 | 0.94 | 0.81 | High TN, low TP, FN remains high | Accuracy is acceptable, but minority image classes are under-detected |
| Image | Random Oversampling | 0.89 | 0.84 | 0.78 | 0.81 | 0.79 | 0.88 | 0.87 | 0.81 | 0.92 | 0.85 | TP increased, FN reduced moderately | Improves minority recall but may increase overfitting risk |
| Image | Random Undersampling | 0.84 | 0.79 | 0.76 | 0.77 | 0.75 | 0.83 | 0.82 | 0.77 | 0.88 | 0.82 | Reduced FN, but TN and total correct predictions decreased | Loss of majority class information reduces overall stability |
| Image | SMOTE | 0.90 | 0.87 | 0.84 | 0.85 | 0.83 | 0.89 | 0.90 | 0.85 | 0.91 | 0.88 | TP improved, FN reduced | Synthetic samples improve minority representation in feature space |
| Image | ADASYN | 0.89 | 0.85 | 0.85 | 0.85 | 0.82 | 0.88 | 0.89 | 0.84 | 0.90 | 0.87 | Minority TP increased, slight FP increase | Adaptive sampling helps difficult minority regions but may add noise |
| Image | Class Weighting | 0.91 | 0.88 | 0.87 | 0.87 | 0.85 | 0.90 | 0.91 | 0.87 | 0.92 | 0.90 | FN reduced with controlled FP | Weighted loss improves minority learning without duplicating samples |
| Image | Focal Loss | 0.91 | 0.89 | 0.88 | 0.88 | 0.86 | 0.91 | 0.92 | 0.88 | 0.92 | 0.90 | Hard minority samples classified more accurately | Focuses learning on difficult examples and improves class discrimination |
| Image | Data Augmentation | 0.92 | 0.90 | 0.89 | 0.89 | 0.87 | 0.91 | 0.93 | 0.89 | 0.93 | 0.91 | TP increased and FN reduced substantially | Visual transformations improve generalization for minority image classes |
| Image | Hybrid Sampling methods | 0.91 | 0.89 | 0.88 | 0.88 | 0.86 | 0.90 | 0.92 | 0.88 | 0.92 | 0.90 | Improved TP with balanced FP control | Combines sample diversity with reduced imbalance effects |
| Image | Data Augmentation + Class Weighting | 0.93 | 0.91 | 0.91 | 0.91 | 0.89 | 0.92 | 0.94 | 0.91 | 0.94 | 0.92 | Highest TP, lowest FN, stable TN | Best placeholder result; improves minority class generalization |

**Best-performing image technique:** Data Augmentation + Class Weighting. This approach is highlighted because it improves minority class recall and Macro F1-Score while maintaining high specificity and weighted performance.

Compared with the baseline, balanced techniques reduce false negatives and increase minority class true positives. Data augmentation is particularly suitable for image data because it produces meaningful visual variation while preserving class semantics.

## 4. Audio Dataset Results

The audio dataset evaluation measures the ability of balancing methods to detect rare acoustic events or minority audio classes. Audio datasets often contain temporal and spectral variation, making augmentation and loss-based approaches suitable for improving minority class robustness.

| Dataset / Modality | Balancing Technique | Accuracy | Precision | Recall | F1-Score | Macro F1-Score | Weighted F1-Score | ROC-AUC | PR-AUC | Specificity | Balanced Accuracy | Confusion Matrix Summary | Key Observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Audio | Baseline without balancing | 0.85 | 0.78 | 0.62 | 0.69 | 0.67 | 0.83 | 0.80 | 0.70 | 0.93 | 0.78 | High FN for rare audio classes | Model favors frequent sound categories |
| Audio | Random Oversampling | 0.86 | 0.80 | 0.74 | 0.77 | 0.74 | 0.85 | 0.84 | 0.76 | 0.91 | 0.82 | TP improved, FN reduced | Improves minority audio recall but may repeat limited samples |
| Audio | Random Undersampling | 0.81 | 0.75 | 0.72 | 0.73 | 0.70 | 0.80 | 0.79 | 0.72 | 0.86 | 0.79 | Better minority TP but lower majority TN | Removes useful majority class acoustic variation |
| Audio | SMOTE | 0.86 | 0.81 | 0.77 | 0.79 | 0.76 | 0.85 | 0.85 | 0.78 | 0.90 | 0.84 | FN reduced, moderate FP increase | Feature-space interpolation improves rare class coverage |
| Audio | ADASYN | 0.86 | 0.80 | 0.79 | 0.79 | 0.77 | 0.85 | 0.86 | 0.79 | 0.89 | 0.84 | More rare-class TP, slight FP increase | Useful for difficult minority audio samples |
| Audio | Class Weighting | 0.87 | 0.83 | 0.80 | 0.81 | 0.78 | 0.86 | 0.87 | 0.81 | 0.90 | 0.85 | Minority FN reduced with stable TN | Improves rare class emphasis during optimization |
| Audio | Focal Loss | 0.89 | 0.85 | 0.84 | 0.84 | 0.81 | 0.88 | 0.90 | 0.84 | 0.91 | 0.87 | Difficult rare sounds correctly classified | Strong improvement for hard-to-classify audio classes |
| Audio | Data Augmentation | 0.88 | 0.84 | 0.83 | 0.83 | 0.80 | 0.87 | 0.89 | 0.83 | 0.90 | 0.86 | TP increased through spectral and temporal variation | Improves robustness to noise, pitch, and temporal shifts |
| Audio | Hybrid Sampling methods | 0.88 | 0.84 | 0.82 | 0.83 | 0.80 | 0.87 | 0.89 | 0.82 | 0.91 | 0.86 | Balanced TP gain and FP control | Sampling combination improves minority representation |
| Audio | Focal Loss + Augmentation | 0.90 | 0.87 | 0.86 | 0.86 | 0.83 | 0.89 | 0.91 | 0.86 | 0.92 | 0.89 | Highest TP for rare audio classes, FN minimized | Best placeholder result; improves rare audio event recognition |

**Best-performing audio technique:** Focal Loss + Augmentation. This combination is suitable because focal loss prioritizes difficult minority examples, while augmentation improves robustness to acoustic variability.

Compared with the baseline, the balanced audio models show substantial improvement in Recall, Macro F1-Score, PR-AUC, and Balanced Accuracy. This indicates improved recognition of rare audio categories rather than merely preserving majority class performance.

## 5. Text Dataset Results

The text dataset evaluation examines whether balancing improves detection of minority labels, such as rare topics, minority sentiments, or low-frequency intents. For text data, class weighting and oversampling are often effective because they can improve minority representation without substantially altering linguistic structure.

| Dataset / Modality | Balancing Technique | Accuracy | Precision | Recall | F1-Score | Macro F1-Score | Weighted F1-Score | ROC-AUC | PR-AUC | Specificity | Balanced Accuracy | Confusion Matrix Summary | Key Observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Text | Baseline without balancing | 0.89 | 0.83 | 0.66 | 0.74 | 0.72 | 0.87 | 0.85 | 0.77 | 0.95 | 0.81 | High TN, many minority FN | Majority text labels dominate predictions |
| Text | Random Oversampling | 0.90 | 0.85 | 0.80 | 0.82 | 0.80 | 0.89 | 0.88 | 0.82 | 0.93 | 0.86 | TP increased, FN reduced | Improves minority label detection with manageable FP increase |
| Text | Random Undersampling | 0.85 | 0.80 | 0.77 | 0.78 | 0.76 | 0.84 | 0.83 | 0.78 | 0.89 | 0.83 | Minority recall improved but majority coverage reduced | Reduced training data weakens text representation learning |
| Text | SMOTE | 0.89 | 0.84 | 0.79 | 0.81 | 0.78 | 0.88 | 0.87 | 0.81 | 0.92 | 0.85 | Minority TP improved | Effective when applied to dense embeddings rather than raw text |
| Text | ADASYN | 0.88 | 0.83 | 0.80 | 0.81 | 0.78 | 0.87 | 0.87 | 0.80 | 0.91 | 0.85 | Rare-label TP improved, FP slightly increased | Helps difficult minority embedding regions but requires noise control |
| Text | Class Weighting | 0.91 | 0.87 | 0.84 | 0.85 | 0.83 | 0.90 | 0.90 | 0.85 | 0.94 | 0.89 | FN reduced while preserving TN | Strong balance between minority recall and overall text classification |
| Text | Focal Loss | 0.90 | 0.86 | 0.83 | 0.84 | 0.82 | 0.89 | 0.89 | 0.84 | 0.93 | 0.88 | Hard minority examples improved | Useful for difficult minority text classes |
| Text | Data Augmentation | 0.90 | 0.85 | 0.82 | 0.83 | 0.81 | 0.89 | 0.88 | 0.83 | 0.93 | 0.87 | TP improved through paraphrased samples | Useful when augmented text preserves label meaning |
| Text | Hybrid Sampling methods | 0.91 | 0.87 | 0.84 | 0.85 | 0.83 | 0.90 | 0.90 | 0.85 | 0.93 | 0.89 | Minority FN reduced with stable FP | Combines representation balance and loss sensitivity |
| Text | Class Weighting + Oversampling | 0.92 | 0.88 | 0.86 | 0.87 | 0.85 | 0.91 | 0.91 | 0.87 | 0.94 | 0.90 | Highest TP, reduced FN, stable TN | Best placeholder result; improves minority intent or label detection |

**Best-performing text technique:** Class Weighting + Oversampling. This approach improves minority class recall and Macro F1-Score while maintaining strong overall performance.

Compared with the baseline, balancing improves the recognition of low-frequency text labels. Macro F1-Score is particularly important for this modality because it prevents majority classes from masking weak minority class performance.

## 6. Tabular Dataset Results

The tabular dataset evaluation focuses on minority class decision boundary improvement. For structured data, synthetic sampling techniques such as SMOTE and ADASYN are commonly effective because they generate additional minority samples in the feature space.

| Dataset / Modality | Balancing Technique | Accuracy | Precision | Recall | F1-Score | Macro F1-Score | Weighted F1-Score | ROC-AUC | PR-AUC | Specificity | Balanced Accuracy | Confusion Matrix Summary | Key Observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Tabular | Baseline without balancing | 0.91 | 0.84 | 0.59 | 0.69 | 0.70 | 0.89 | 0.83 | 0.72 | 0.97 | 0.78 | Very high TN, high FN | Accuracy is high but minority class detection is weak |
| Tabular | Random Oversampling | 0.90 | 0.82 | 0.78 | 0.80 | 0.79 | 0.89 | 0.87 | 0.81 | 0.93 | 0.86 | TP improved, FN reduced | Better minority detection but duplicate samples may overfit |
| Tabular | Random Undersampling | 0.86 | 0.78 | 0.79 | 0.78 | 0.76 | 0.85 | 0.84 | 0.78 | 0.89 | 0.84 | FN reduced, TN decreased | Majority class information loss reduces overall performance |
| Tabular | SMOTE | 0.92 | 0.87 | 0.85 | 0.86 | 0.84 | 0.91 | 0.91 | 0.86 | 0.94 | 0.90 | TP substantially increased, FN reduced | Strong improvement in minority decision boundary |
| Tabular | ADASYN | 0.91 | 0.85 | 0.86 | 0.85 | 0.84 | 0.90 | 0.91 | 0.86 | 0.93 | 0.90 | Minority TP improved, slight FP increase | Effective for difficult minority regions |
| Tabular | Class Weighting | 0.90 | 0.84 | 0.82 | 0.83 | 0.81 | 0.89 | 0.89 | 0.83 | 0.93 | 0.87 | FN reduced while preserving specificity | Useful for cost-sensitive tabular classifiers |
| Tabular | Focal Loss | 0.90 | 0.85 | 0.83 | 0.84 | 0.82 | 0.89 | 0.90 | 0.84 | 0.93 | 0.88 | Difficult minority samples improved | Effective for models trained with differentiable loss functions |
| Tabular | Data Augmentation | 0.89 | 0.82 | 0.80 | 0.81 | 0.79 | 0.88 | 0.87 | 0.81 | 0.92 | 0.86 | Minority TP improved moderately | Requires careful feature-aware augmentation |
| Tabular | Hybrid Sampling methods | 0.92 | 0.88 | 0.86 | 0.87 | 0.85 | 0.91 | 0.92 | 0.87 | 0.94 | 0.90 | TP increased with stable FP | Combines minority synthesis and majority noise reduction |
| Tabular | SMOTE + Tomek Links | 0.93 | 0.89 | 0.87 | 0.88 | 0.86 | 0.92 | 0.93 | 0.88 | 0.95 | 0.91 | Highest TP, reduced FN, cleaner decision boundary | Best placeholder result; improves boundary separation |

**Best-performing tabular technique:** SMOTE + Tomek Links. This hybrid approach is suitable for tabular data because it increases minority class representation while removing ambiguous majority-minority boundary samples.

Compared with the baseline, tabular balancing techniques substantially improve Recall, PR-AUC, Macro F1-Score, and Balanced Accuracy. This confirms that high baseline accuracy was mainly influenced by the majority class.

## 7. Cross-Modality Comparative Analysis

Across all four modalities, balancing techniques consistently improve minority class prediction. The most meaningful improvements are observed in Recall, Macro F1-Score, PR-AUC, and Balanced Accuracy. These metrics show whether the model is learning minority class patterns rather than simply optimizing for majority class accuracy.

| Dataset / Modality | Baseline Accuracy | Best Balanced Accuracy | Baseline Macro F1-Score | Best Macro F1-Score | Baseline PR-AUC | Best PR-AUC | Best Technique | Main Improvement |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Image | 0.88 | 0.92 | 0.71 | 0.89 | 0.76 | 0.91 | Data Augmentation + Class Weighting | Improved visual minority class generalization |
| Audio | 0.85 | 0.89 | 0.67 | 0.83 | 0.70 | 0.86 | Focal Loss + Augmentation | Better detection of rare audio events |
| Text | 0.89 | 0.90 | 0.72 | 0.85 | 0.77 | 0.87 | Class Weighting + Oversampling | Improved minority label and intent detection |
| Tabular | 0.91 | 0.91 | 0.70 | 0.86 | 0.72 | 0.88 | SMOTE + Tomek Links | Improved minority decision boundary separation |

The results indicate that the most appropriate imbalance handling method depends on the data modality. Image and audio data benefit strongly from augmentation because transformations can preserve class identity while increasing diversity. Text data benefits from class weighting and oversampling, provided that semantic meaning is preserved. Tabular data benefits from synthetic sampling and hybrid boundary-cleaning methods that improve minority class separation in structured feature space.

## 8. Best Performing Technique Per Dataset

| Dataset | Best Technique | Reason |
|---|---|---|
| Image | Data Augmentation + Class Weighting | Improved minority class generalization by increasing visual diversity and reducing majority class dominance during training |
| Audio | Focal Loss + Augmentation | Better handling of rare audio classes by emphasizing difficult examples and improving robustness to acoustic variation |
| Text | Class Weighting / Oversampling | Improved minority intent or label detection while maintaining strong majority class performance |
| Tabular | SMOTE / ADASYN | Improved minority class decision boundary by generating synthetic samples in underrepresented feature regions |

## 9. Key Findings

1. Accuracy alone is not sufficient for evaluating imbalanced classification because high accuracy can be achieved even when minority class recall is poor.
2. Balanced Accuracy, Macro F1-Score, Recall, ROC-AUC, and PR-AUC provide more reliable evidence of improved minority class prediction.
3. Random oversampling improves minority recall but may increase overfitting when the same minority samples are repeatedly presented during training.
4. Random undersampling can reduce majority class bias, but it may discard useful majority class information and reduce overall model stability.
5. SMOTE and ADASYN are especially suitable for tabular and embedding-based representations because they improve minority class coverage in feature space.
6. Class weighting and focal loss are effective loss-level strategies because they improve minority class sensitivity without directly altering the dataset distribution.
7. Data augmentation is most effective for image and audio modalities when transformations preserve class semantics and increase meaningful sample diversity.

## 10. Conclusion

The application of class imbalance handling techniques improves the fairness, robustness, and minority class recognition capability of multimodal classification models. Across image, audio, text, and tabular datasets, balanced techniques reduce false negatives, increase minority class true positives, and improve Macro F1-Score, PR-AUC, ROC-AUC, and Balanced Accuracy compared with baseline models.

The comparative analysis shows that no single balancing technique is universally optimal across all modalities. Image datasets benefit most from Data Augmentation combined with Class Weighting, audio datasets benefit from Focal Loss combined with Augmentation, text datasets benefit from Class Weighting and Oversampling, and tabular datasets benefit from SMOTE, ADASYN, and hybrid sampling methods. These findings support the use of modality-aware imbalance handling strategies for building more reliable, fair, and generalizable multimodal classification systems.

All metric values in this document are placeholders and must be replaced with actual experimental results after final model training, validation, and testing.
