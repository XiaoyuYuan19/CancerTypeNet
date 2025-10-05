# CancerTypeNet: Tumor Type Classification using Mutational Signatures

## Team Members
**Equal Contribution:** Johanna Aspholm, Hanlu Li, Xiaoyu Yuan, Mudong Guo, Hao Liu Emilia Miettinen, Jami Heljomaa,

---

## Project Overview

This project implements multiple machine learning and deep learning models to classify cancer types based on mutational signatures and mutational catalogs. Inspired by the work of [Alexandrov et al. (2020)](https://www.nature.com/articles/s41586-020-1943-3) on the repertoire of mutational signatures in human cancer, we explore various classification approaches to predict cancer types from genomic mutation patterns.

### Key Objectives
- Compare mutational profiles (96-channel trinucleotide contexts) with mutational signature activities for cancer type prediction
- Implement and evaluate 6 different classification models
- Analyze feature importance across mutation channels
- Handle class imbalance in cancer type distributions
- Achieve interpretable results for cancer genomics research

---

## Dataset

### Data Structure

The dataset consists of mutational catalogs and predicted signature activities from both **Whole Genome Sequencing (WGS)** and **Whole Exome Sequencing (WES)** data:

#### 1. Mutational Catalogs
- **Format**: 96 trinucleotide mutation channels (6 mutation types × 16 trinucleotide contexts)
- **Mutation Types**: C>A, C>G, C>T, T>A, T>C, T>G
- **Structure**: Each sample represented as a vector of mutation counts across 96 channels

#### 2. Signature Activities  
- **Format**: Activities of 65 known mutational signatures (SBS1-SBS65)
- **Accuracy Score**: Cosine similarity between reconstructed and observed catalogs

#### 3. Data Sources
- **WGS_PCAWG**: Whole genome data from Pan-Cancer Analysis of Whole Genomes
- **WGS_Other**: Additional whole genome datasets
- **WES_TCGA**: Exome data from The Cancer Genome Atlas
- **WES_Other**: Additional exome datasets

#### 4. Cancer Types
37 cancer types with varying sample sizes (e.g., Liver-HCC, Kidney-RCC, Breast-AdenoCA, etc.)

**Note**: WES data is normalized to sum to 1 due to the smaller genomic coverage (~1% of genome) compared to WGS.

### Data Distribution

<img width="1189" height="989" alt="image" src="https://github.com/user-attachments/assets/771b1dd2-8fd6-473a-a4ec-4f84644c87a1" />

*Cell: The cell with `fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))` showing cancer type distributions*

---

## Exploratory Data Analysis

### Mutation Load Analysis

We analyzed the total mutation counts across cancer types for different datasets:

**[INSERT FIGURE: PCAWG WGS mutation load scatter plots]**
*Cell: The cell with `_, ax = plt.subplots(nrows = len(sum_dict)//3+1, ncols = 3, figsize = (16, 5*len(sum_dict)/(3+1)))` for PCAWG WGS*

**[INSERT FIGURE: Other WGS mutation load scatter plots]**
*Cell: Similar plot for non-PCAWG WGS data*

**[INSERT FIGURE: TCGA WES mutation load scatter plots]**
*Cell: Similar plot for TCGA WES with log scale*

Key findings:
- WGS samples show higher mutation counts than WES samples
- Significant variability in mutation load across cancer types
- Some cancer types (e.g., Melanoma) show characteristically high mutation rates

---

## Models Implemented

### 1. Random Forest Classifier

**Method**: Ensemble of decision trees with bootstrap aggregation and random feature selection.

**Performance**:
- **PCAWG WGS**: 72% accuracy
- **Other WGS**: 65% accuracy  
- **TCGA WES**: 58% accuracy
- **Other WES**: 52% accuracy

<img width="511" height="415" alt="image" src="https://github.com/user-attachments/assets/28bc007e-e955-495f-832f-fceeb60bbe3d" />

*Cell: `ax = sns.heatmap(confusion_matrix(y_test_pcawg_wgs, model_pcawg_wgs.predict(X_test_pcawg_wgs)), annot = True)`*

**Feature Importance Analysis**:
- C>T mutation channel identified as most informative
- T>C channel also shows high predictive power
- All 96 channels contribute to optimal performance

---

### 2. Support Vector Machine (SVM)

**Method**: Maximum-margin classifier with StandardScaler preprocessing pipeline.

**Performance**:
| Dataset | Accuracy |
|---------|----------|
| Catalog WGS PCAWG | 67.6% |
| Catalog WES TCGA | 49.5% |
| Activity WGS PCAWG | 60.0% |
| Activity WES TCGA | 51.9% |

**Key Insights**:
- Performance heavily dependent on hyperparameter tuning (not fully optimized)
- Benefits from stratified train-test split
- Works better with normalized features

---

### 3. Deep Neural Network (DNN)

**Architecture**:
- Input layer: 96 features (mutational channels) or 161 features (combined)
- Hidden layers: 128 → 64 → output
- Dropout: 0.5 (regularization)
- Activation: ReLU
- Output: Softmax (multi-class classification)
- Optimizer: Adam (lr=0.001)
- Loss: Sparse categorical cross-entropy

**Optimal Hyperparameters** (from grid search):
- Learning rate: 0.001
- Batch size: 32
- Epochs: 2000

**Performance**:
- **WGS_PCAWG Catalog**: 65.77% validation accuracy
- **WGS_PCAWG Activity**: ~62% validation accuracy
- **Combined (Catalog + Activity)**: Improved performance

**[INSERT FIGURE: Training history plots showing accuracy and loss curves]**
*Cell: From `plot_history()` function showing training and validation curves*

**[INSERT FIGURE: Confusion matrix with F1 and Recall scores]**
*Cell: From `plot_ax()` function showing detailed classification metrics*

**34→21 Class Reduction**: Merging subtypes (e.g., Bone-Benign, Bone-Epith → Bone) improved interpretability with minimal accuracy loss.

---

### 4. Convolutional Neural Network (CNN)

**Architecture**:
- 3 Convolutional layers (feature extraction)
- 5 Linear layers (classification)
- Dropout layers for regularization
- Input: 96 or 161 features
- Output: 37 cancer types

**Key Features**:
- Automatically learns hierarchical mutation patterns
- No manual feature engineering required
- Cross-entropy loss with Adam optimizer

**Performance**: 
- Achieved comparable performance to DNN
- Better at capturing local mutation patterns

**[INSERT FIGURE: CNN confusion matrix]**
*Cell: From the CNN testing section showing the heatmap*

---

### 5. K-Nearest Neighbors (KNN)

**Method**: Instance-based learning with Median-MAD normalization.

**Key Experiments**:

#### Mutation Type Analysis
- **Dropping C>T**: Lowest accuracy drop → least biased channel
- **Keeping only C>T**: Best single-channel accuracy (23.7%)
- **Keeping only T>C**: Second-best single-channel accuracy (21.6%)

**[INSERT FIGURE: Confusion matrix for C>T channel prediction]**
*Cell: From `plot_cm_of_CtoT()` function*

**[INSERT FIGURE: Confusion matrix for T>C channel prediction]**  
*Cell: From `plot_cm_of_TtoC()` function*

#### Trinucleotide Context Analysis
- Systematic analysis of all 16 trinucleotide contexts
- Identified contexts with highest/lowest predictive power

**Takeaway**: C>T mutations show the least bias and highest individual predictive power in the training dataset.

---

### 6. Gradient Boosting Classifier

**Method**: Sequential ensemble of weak decision tree learners with gradient descent optimization.

**Performance**:
| Dataset | Accuracy |
|---------|----------|
| Catalog WGS PCAWG | 64.7% |
| Catalog WES TCGA | 49.0% |
| Activity WGS PCAWG | 60.0% |
| Activity WES TCGA | 52.0% |
| **Combined WGS** | **69.5%** ✨ |
| **Combined WES** | **54.2%** |

**Feature Importance (96 channels)**:

<img width="1626" height="600" alt="image" src="https://github.com/user-attachments/assets/07bf7855-21c8-4951-9c98-6be02a90c57d" />

*Cell: The cell with `plt.bar(feature_df['Feature'], feature_df['Importance'])` showing all 96 channels*

**Feature Importance (6 mutation types)**:

<img width="851" height="560" alt="image" src="https://github.com/user-attachments/assets/357f1051-2666-4211-b634-51dfc5bf3b93" />

*Cell: The cell showing reduced 6-channel feature importance after grouping*

**Key Findings**:
- C>T channel is the most crucial
- T>A channel is the least important
- Combining catalogs and activities improves accuracy by ~5%

---

## Model Comparison & Results

### Best Performance Summary

| Model | Best Dataset | Test Accuracy | Key Advantage |
|-------|-------------|---------------|---------------|
| **Deep Neural Network** | WGS Combined | **69.5%** | Regularization, complex patterns |
| **Gradient Boosting** | WGS Combined | 69.5% | Feature importance, robustness |
| Random Forest | WGS PCAWG | 72% | Interpretability, no tuning |
| CNN | WGS Combined | ~65% | Automatic feature learning |
| SVM | WGS PCAWG | 67.6% | Margin-based classification |
| KNN | WGS PCAWG | ~50% | Simple, instance-based |

### Key Insights

1. **Data Quality Matters**: WGS data consistently outperforms WES data due to higher mutation counts and genomic coverage.

2. **Feature Combination**: Combining mutational channels with signature activities improves accuracy by 3-5% across most models.

3. **Class Imbalance**: Cancer types with <15 samples showed poor prediction accuracy. Filtering these improved overall model performance.

4. **Deep Learning Advantage**: DNNs with dropout regularization achieve the best balance between accuracy and generalization.

5. **C>T Dominance**: Across all models, C>T mutations emerged as the most informative feature for cancer type prediction.

---

## Data Preprocessing

### Key Steps
1. **Cancer type extraction**: Parsed from sample names (format: `CancerType::SampleID`)
2. **Class filtering**: Removed cancer types with <5 samples
3. **Normalization**: 
   - StandardScaler (mean=0, std=1) for DNN/SVM
   - Median-MAD normalization for KNN
   - WES data normalized to sum=1 for comparability with WGS
4. **Train-test split**: 80-20 ratio with stratification by cancer type
5. **Label encoding**: Converted cancer type strings to integer labels

---

## Installation & Requirements

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Deep learning
pip install torch tensorflow keras

# Visualization
pip install plotly
```

**Python Version**: 3.8+

---

## Usage

```python
# Load data
import pandas as pd
catalogs = pd.read_csv("path/to/WGS_PCAWG.96.csv")
activities = pd.read_csv("path/to/WGS_PCAWG.activities.csv")

# Preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Train model (example: Random Forest)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

See notebooks for detailed implementation of each model.

---

## References

1. Alexandrov, L.B., et al. (2020). [The repertoire of mutational signatures in human cancer](https://www.nature.com/articles/s41586-020-1943-3). *Nature*.

2. Jiao, W., et al. (2019). [A deep learning system accurately classifies primary and metastatic cancers using passenger mutation patterns](https://www.nature.com/articles/s41467-019-13825-8). *Nature Communications*.

3. Pan-Cancer Analysis of Whole Genomes Consortium. [Nature Collection](https://www.nature.com/collections/afdejfafdb).

---

## License

This project is for academic purposes. Data sourced from PCAWG and TCGA public repositories.

---

## Contact

For questions about this project, please open an issue in the repository.

**Last Updated**: October 2025
