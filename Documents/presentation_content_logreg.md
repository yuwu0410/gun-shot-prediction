# Presentation Content: Logistic Regression Models

---

## 5. Implementation (20%)

**Did we implement multiple models?**
No baseline (maybe can add one later, like multi-class). We implemented an ensemble of three binary Logistic Regression models (One-vs-All strategy) with SMOTE. Each binary model is a specialist (Suicide vs Rest, Homicide vs Rest, Others vs Rest), then we combine them via max-probability voting for final prediction. Should have a baseline without SMOTE to show improvement. This is a major limitation.

**Are they correctly implemented?**
Yes. Split train/test (80/20) before SMOTE, preventing data leakage. Ensemble structure: train three binary specialists, each with independent SMOTE. Final prediction uses max-probability voting across the three models. One-hot encoding with drop_first avoids multicollinearity. Merged Accidental + Undetermined into "Others" (4-class to 3-class).

**Did we tune them?**
No. We set max_iter=1000 for convergence. Didn't tune C (regularization), penalty (L1/L2), solver, or class_weight. No validation set means no way to compare configs.(Later could re-split the dataset to have a validation set)

---

## 6. Model Evaluation (18%)

**Macroscopic (dataset-wide) performance**
Overall accuracy: 78.2% on test set (19,603 samples). Weighted avg F1: 80% accounts for class sizes. Macro avg F1: 56% reveals performance drops significantly when treating all classes equally. The gap between weighted (80%) and macro (56%) F1 shows the model performs well on majority classes but struggles with minority class.

**Microscopic performance**
Suicide (12,473 samples, 64%): 87% precision, 81% recall, 84% F1. Best as largest class.
Homicide (6,656 samples, 34%): 76% precision, 78% recall, 77% F1. Good as second-largest.
Others (474 samples, 2.4%): 5% precision, 14% recall, 8% F1. Still poor.

Binary models: Suicide 83% accuracy, Homicide 84% accuracy, Others 71% accuracy(it's quite good here). But the Others model only catches 210/474 cases (44% recall), underpredicting heavily.

"Feature Importance" coefficients: Sex_Male (+0.811), Race_White (+0.448) predict Suicide. Age (+0.029) is weak despite EDA showing clear age patterns (Suicide median 51, Homicide median 29).

**Improvement demonstration**
(To be done) None. No baseline means can't show if SMOTE helped or if 78.2% is good. 

---

## 7. Results Interpretation (10%)

**Error analysis**
Others fails (5% precision, 14% recall) for three reasons:
(a), Only 474 samples vs 12,473 Suicide (26:1). Model sees Suicide 26x more, learns it perfectly but not Others.
(b), Merging Accidental + Undetermined is a trade-off: 
Benefits: combining gives ~474 total samples instead of 241+233 separately, simplifies to 3-class problem. 
Drawbacks: they likely have different patterns - Accidental involves mishaps (different age/location distributions), Undetermined has missing information (different feature missingness). Merging blurs these distinctions, making the combined "Others" even harder to learn.
(c) Features have severe overlap across classes. Same demographic profile (e.g., 37-year-old white male at home) appears in all three classes ("bad" data for model to learn), making them indistinguishable with only age/sex/race/location.

**Why it works for major classes**
78.2% ensemble accuracy is reasonable given 26:1 imbalance. Key factors: 
(1) OvA ensemble - each binary specialist focuses independently with clearer decision boundaries than multiclass.
(2) SMOTE prevents total minority failure - 14% recall catches 1 in 7 Others cases.
(3) Feature importance & class alignment - Sex_Male (+0.811) matches 63% males in Suicide, Race_White (+0.448) matches 85% whites. 
(4) Sufficient samples - Suicide/Homicide models hit 83-84% with adequate training data.

**Future improvements**
Add baseline multi-class Logistic Regression without SMOTE to prove SMOTE's value. 
Add validation set for tuning. Tune C, penalty, solver, class_weight with GridSearchCV. Try 4-class problem keeping Accidental and Undetermined separate, or use hierarchical classification. 
Engineer better features: time-of-day, weapon type, location details. Try weighted voting or stacking ensemble.

---

