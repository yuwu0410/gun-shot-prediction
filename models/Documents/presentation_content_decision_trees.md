# Presentation Content: Decision Tree Related Models & Methods

---

## 1. Originality (4%)

**Five-model systematic comparison**
We implemented five different models to tackle the 42:1 class imbalance: baseline Decision Tree, Decision Tree with class weighting, Decision Tree with SMOTE, Random Forest, and XGBoost. This systematic comparison from simple to complex shows what actually works for extreme imbalance in our gun incident prediction task.

**Preventing data leakage**
We used imbalanced-learn Pipeline so SMOTE is applied inside each CV fold, not before splitting. This is critical because applying oversampling before cross-validation leaks information and artificially inflates performance.

**Comprehensive hyperparameter tuning**
Each model went through GridSearchCV with 72-90 hyperparameter combinations using F1-weighted scoring. We optimized multiple parameters like max_depth, min_samples_split/leaf, criterion for trees, and n_estimators, learning_rate for ensemble methods.

---

## 2. Relevance (4%)

**Core course concepts applied**
Decision Trees with class weight and entropy splitting criteria. Data processing and feature engineering with StandardScaler and OneHotEncoder, turning 9 raw features into 26 engineered features. GridSearchCV with 5-fold cross-validation for validation. Regularization through tree depth and sample requirements.

**Evaluation for imbalanced data**
We used F1-weighted instead of accuracy as our main metric, plus per-class precision, recall, and confusion matrices - exactly what we learned for handling imbalanced datasets.

**Bias-variance tradeoff**
The hyperparameter tuning directly addresses the bias-variance tradeoff: deeper trees overfit (high variance), shallower trees underfit (high bias). Grid search finds the balance.

---

## 3. Related Work (4%)

**SMOTE oversampling**
We used SMOTE from "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla, Bowyer, Hall, and Kegelmeyer. It creates synthetic minority samples by interpolating between existing ones instead of just duplicating. We implemented it through the imbalanced-learn library.

**Ensemble methods**
Random Forest and XGBoost are well-established ensemble approaches. Random Forest uses bagging with multiple decision trees. XGBoost from "XGBoost: A Scalable Tree Boosting System" by Chen and Guestrin uses gradient boosting and has built-in handling for imbalanced data.

**Our novel contribution**
We use iterative model selection strategy. Start with baseline Decision Tree to understand the problem. Then test two approaches to handle imbalance: class weighting (algorithm-level) and SMOTE (data-level). This comparison shows SMOTE works better. Next, we move to ensemble methods - Random Forest adds bagging, XGBoost adds gradient boosting. Each step builds on insights from previous models. This iterative approach reveals that even with sophisticated methods, minority classes (undetermined + accidentally) with 100-200 samples hit fundamental limits.

---

## 4. Technical Justification (10%)

**Is the approach suitable and valid?**
Our approach is well-suited for this 4-class extreme imbalance problem. We use tree-based models because they handle non-linear relationships and mixed feature types naturally. The 42:1 class ratio makes this hard, so we tested multiple strategies: class weighting (algorithm-level), SMOTE (data-level), and ensemble methods (model-level). Each addresses imbalance differently, letting us find what actually works.

**Are there technical flaws?**
No major flaws. We use imbalanced-learn Pipeline to apply SMOTE inside each CV fold, preventing data leakage. The proper train-validation-test split (70-15-15) ensures unbiased evaluation. All models use 5-fold cross-validation for reliable hyperparameter selection.

**Are metrics appropriate and correctly interpreted?**
Yes. We use F1-weighted as the main metric during training because it accounts for class imbalance better than accuracy. For evaluation, we report accuracy, precision, recall, and F1 per class plus macro/weighted averages. Confusion matrices show where models make mistakes. Feature importance plots reveal which features drive predictions. This gives both dataset-wide performance and class-specific insights.

---

## 5. Implementation (20%)

**Did we implement multiple models?**
Yes. Five models from simple to complex: Baseline Decision Tree (tests deeper trees 15-30), Weighted Decision Tree (tests wider range including unlimited depth), SMOTE Decision Tree (oversampling), Random Forest (bagging ensemble), and XGBoost (gradient boosting). Baseline is simplest, XGBoost is best performer.

**Are they correctly implemented?**
All models use scikit-learn and imbalanced-learn standard implementations. Each training script loads preprocessed data, defines parameter grids, runs GridSearchCV with 5-fold CV and F1-weighted scoring, selects the best model, and saves it. For SMOTE models, we use Pipeline to ensure proper cross-validation without leakage. XGBoost needs integer labels not strings, so we use LabelEncoder to convert 'accidental'→0, 'homicide'→1, etc., and save the encoder to decode predictions back to class names during evaluation.

**Did we tune them?**
Extensively. Baseline DT: 72 combinations focusing on deeper trees (max_depth 15-30). Weighted DT: 90 combinations with wider range including unlimited depth. Random Forest: 72 combinations adding n_estimators for ensemble diversity. XGBoost: 48 combinations tuning gradient boosting specific parameters like learning_rate and subsample. Parameter ranges are chosen based on problem complexity - we test both conservative (shallow trees, high regularization) and aggressive (deep trees, low regularization) settings to find the sweet spot.

---

## 6. Model Evaluation (18%)

**Macroscopic (dataset-wide) performance**
XGBoost achieves best performance: 73.9% accuracy, 73.3% F1-weighted. Random Forest second at 69.8%. Decision Tree variants range 56-64%. All evaluated on held-out test set that was never used during training or tuning.

**Microscopic (per-class) performance**
Result in Summary Table -> XGBoost shows clear patterns. Major classes succeed: Suicide (75% precision, 83% recall), Homicide (76%, 68%). Minority classes fail: Accidental (11%, 6%), Undetermined (2%, 2%). 
Feature importance Graph -> reveals age, place_home, sex_male as top predictors - demographic/location features shared across all classes. 
Confusion matrix -> 94% of accidental cases misclassified as suicide/homicide due to overlapping features (age 20-60, same locations). With only 241 accidental samples, the model can't learn distinguishing patterns.

**Improvement demonstration**
Clear improvement from baseline to best => Baseline Decision Tree: 57.1%. SMOTE: 63.7% (+6.6%). Random Forest: 69.8% (+12.7%). XGBoost: 73.9% (+16.8%). Progressive improvement with XGBoost showing largest gain.

**Why performance isn't higher**
For minority class: SMOTE with 121-241 samples creates variations, not true diversity. Class weighting can't create missing information. 
Core issue: feature overlap - accidental shootings share age/location patterns with other types. The 42:1 imbalance means models learn majority boundaries from thousands of examples but lack data to distinguish minority classes. Standard ML techniques hit fundamental limits here. We will talk about more advanced approaches in future work part.

---

## 7. Results Interpretation (10%)

**Error analysis with evidence**
Three root causes for minority class failure. 
First, sample scarcity: 241 accidental vs 121 undetermined out of 62,267 total (210x imbalance). 
Second, feature overlap: our 26 features (age, sex, race, place, education) lack discriminative signals - a 30-year-old male at home could be any class. 
Third, rational prediction bias: with 6% recall for accidental, the model defaults to majority classes given prior probability.

**Why XGBoost succeeds**
73.9% accuracy from gradient boosting's iterative error correction. 150 trees with max_depth=7 and learning_rate=0.1 balance model complexity. SMOTE in pipeline prevents overfitting. The +16.8% over baseline proves it works despite data limits.

**Future improvements**
(a), Collect more minority class data - even 1,000 accidental cases would help. Engineer domain-specific features like time-of-day, weapon-type, or witness-count that might discriminate better than demographics. 
(b), Try cost-sensitive learning with custom loss functions to penalize minority errors more heavily than simply using class_weight. 
(c), Explore ensemble combinations - stacking XGBoost with Random Forest might reduce errors. 
(d), Consider treating it as Anomaly Detection for minority classes rather than standard classification, since they're so rare.

---
