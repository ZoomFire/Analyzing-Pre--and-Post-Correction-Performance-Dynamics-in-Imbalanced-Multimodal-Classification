# Comparative Analysis of Class Imbalance Handling Techniques Across Multi-Modal Datasets

This repository contains the implementation and documentation for the project:

**Comparative Analysis of Class Imbalance Handling Techniques Across Multi-Modal Datasets: Image, Audio, Text, and Tabular Data**

The project evaluates how different class imbalance handling techniques affect model performance across four data modalities: image, audio, text, and tabular data. The main focus is to compare baseline model performance with balanced model performance using evaluation metrics that are more reliable than accuracy for imbalanced classification problems.

## Project Overview

Class imbalance is a common problem in real-world machine learning datasets where one or more classes contain significantly fewer samples than others. In such cases, models may achieve high accuracy while still failing to correctly identify minority class samples.

This project studies the effect of several imbalance handling techniques, including:

- Random Oversampling
- Random Undersampling
- SMOTE
- ADASYN
- SMOTEENN
- SMOTETomek
- Class Weighting
- Focal Loss
- MixUp
- CutMix
- Data Augmentation
- Hybrid Sampling Methods

The evaluation emphasizes minority class performance, fairness, and robustness across multiple dataset types.

## Dataset Modalities

The project is organized around four dataset categories:

| Modality | Objective |
|---|---|
| Image | Evaluate imbalance handling for visual classification tasks |
| Audio | Evaluate rare-class recognition in audio classification |
| Text | Evaluate minority label or intent detection in text classification |
| Tabular | Evaluate minority class prediction in structured data, such as fraud detection |

## Evaluation Metrics

The project uses multiple metrics because accuracy alone is not sufficient for imbalanced datasets.

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Reliability of positive predictions |
| Recall | Ability to identify actual minority class samples |
| F1-Score | Balance between Precision and Recall |
| Macro F1-Score | Equal-weighted class performance |
| Weighted F1-Score | Class-support-weighted performance |
| ROC-AUC | Class separation capability |
| PR-AUC | Minority-positive prediction quality |
| Specificity | Correct identification of negative or majority samples |
| Balanced Accuracy | Average performance across classes |
| Confusion Matrix | Detailed TP, TN, FP, and FN behavior |

## Repository Structure

```text
.
├── analysis.py
├── balanced_techniques_evaluation_metrics.md
├── config.py
├── models/
│   └── resnet.py
├── src/
│   ├── phase1/
│   ├── phase1_baseline/
│   ├── phase2/
│   ├── phase2_improved_model/
│   ├── phase3/
│   ├── phase3_balancing/
│   ├── phase4/
│   ├── phase4_evaluation/
│   └── text/
├── text-module/
│   ├── main.py
│   └── src/
├── *.csv
├── *.png
├── README.md
└── requirements.txt
```

## Project Phases

### Phase 1: Baseline Model Development

Baseline models are trained on the original imbalanced datasets. This phase establishes the initial performance before applying correction techniques.

### Phase 2: Improved Model Development

Improved architectures and feature engineering methods are used to strengthen the baseline models before imbalance correction.

### Phase 3: Class Imbalance Handling

Multiple balancing techniques are applied and compared. The tabular pipeline includes methods such as Random Oversampling, Random Undersampling, SMOTE, ADASYN, SMOTEENN, and SMOTETomek.

### Phase 4: Evaluation and Visualization

Final evaluation includes metric calculation, confusion matrix analysis, comparison graphs, ROC curves, PR curves, and result documentation.

## Key Documentation File

The file below contains a structured research-style explanation of evaluation metrics after applying class imbalance handling techniques:

```text
balanced_techniques_evaluation_metrics.md
```

It includes modality-wise metric tables, baseline versus balanced comparison, best-performing techniques, key findings, and conclusion.

## Setup Instructions

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is incomplete, install the commonly used dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn torch torchvision tqdm
```

## Running the Project

Run the tabular balancing pipeline:

```bash
python -m src.phase3_balancing.main
```

Run the Phase 4 evaluation pipeline:

```bash
python -m src.phase4_evaluation.main
```

Run the general analysis script:

```bash
python analysis.py
```

Run the text module:

```bash
python text-module/main.py
```

## Expected Dataset Placement

Some pipelines expect datasets in the following locations:

```text
datasets/tabular/creditcard.csv
data/
datasets/
```

Large datasets are not included in the repository. Place the required datasets manually before running the training or evaluation scripts.

## Outputs

The project generates evaluation outputs such as:

- Classification reports
- Confusion matrices
- Precision-Recall curves
- ROC curves
- Metric comparison tables
- Training and validation plots

Some existing result files are included in the repository, such as:

```text
comparison.csv
final_results.csv
final_comparison_results.csv
phase3_results.csv
confusion_matrix.png
pr_curve.png
```

## Important Notes

- Large model checkpoints such as `.pth`, `.pt`, `.ckpt`, `.onnx`, and similar files are ignored by Git.
- Virtual environments, datasets, generated outputs, and local cache files are also ignored.
- Placeholder metric values in documentation should be replaced with actual experimental results after final training and evaluation.
- For imbalanced datasets, Macro F1-Score, Recall, PR-AUC, ROC-AUC, and Balanced Accuracy should be prioritized over Accuracy.

## Best Performing Techniques by Modality

| Dataset | Recommended Technique | Reason |
|---|---|---|
| Image | Data Augmentation + Class Weighting | Improves minority class generalization and visual diversity |
| Audio | Focal Loss + Augmentation | Improves detection of rare and difficult audio classes |
| Text | Class Weighting + Oversampling | Improves minority label and intent detection |
| Tabular | SMOTE / ADASYN / SMOTETomek | Improves minority class decision boundaries |

## Research Significance

This project demonstrates that class imbalance handling improves minority class recognition and provides a more reliable evaluation of machine learning models across different data types. The comparative multimodal design helps identify which techniques are most suitable for each modality instead of assuming that one balancing method works equally well for all datasets.

## Conclusion

The project shows that imbalance correction techniques can improve fairness, minority class prediction, and classification robustness. Across image, audio, text, and tabular data, the most effective approach depends on the characteristics of the modality, the model architecture, and the evaluation objective.

This repository is suitable for academic project submission, research documentation, and further experimentation on class imbalance handling in multimodal machine learning.
