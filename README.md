# Gun Shot Incident Prediction
## 枪击事件类型预测 - 机器学习项目

---

## 🎯 Project Overview | 项目概述

**目标 (Objective)**: 基于事件特征预测枪击事件类型 (suicide, homicide, accidental, undetermined)

**关键挑战 (Key Challenge)**: 极度类别不平衡 - 最大类与最小类比例达 **42:1**

**数据规模 (Dataset)**:
- 总样本数: **62,267** 条记录
- 特征数: **26** 个 (5个数值 + 21个类别特征)
- 类别分布: Suicide 54.4% | Homicide 41.7% | Accidental 2.6% | Undetermined 1.3%

---

## ✅ Project Status | 完成状态

### 已完成工作

- ✅ **Data Preprocessing** | 数据预处理 (`notebooks/01_preprocessing.ipynb`)
  - 特征工程 + 标准化
  - Train/Valid/Test split: **70% / 15% / 15%**
  
- ✅ **Model Training** | 模型训练 (5个模型)
  - Decision Tree (Baseline) - GridSearchCV (72 combinations)
  - Decision Tree (Weighted) - class_weight='balanced'
  - Decision Tree (SMOTE) - 过采样少数类
  - Random Forest - 集成学习
  - **XGBoost** - 梯度提升树 ⭐

- ✅ **Model Evaluation** | 模型评估 (Test Set)
  - Performance metrics: Accuracy, Precision, Recall, F1
  - Visualization: Confusion Matrix + Feature Importance
  - Comparison: 5 models 性能对比

---

## 🏆 Key Results | 核心结果

### 最佳模型: XGBoost ⭐

| Model | Accuracy | Recall (Weighted) | F1 (Weighted) | F1 (Macro) |
|-------|----------|-------------------|---------------|------------|
| **XGBoost** | **73.9%** | **73.9%** | **73.3%** | **40.1%** |
| Random Forest | 69.8% | 69.8% | 71.3% | 40.0% |
| Decision Tree (SMOTE) | 63.7% | 63.7% | 66.1% | 36.6% |
| Decision Tree (Baseline) | 57.1% | 57.1% | 61.7% | 34.0% |
| Decision Tree (Weighted) | 56.4% | 56.4% | 62.0% | 36.5% |

### 关键发现 (Key Findings)

✅ **Performance Improvement**: XGBoost 比基础决策树提升 **+16.8%** accuracy

✅ **Major Classes**: Suicide & Homicide 预测准确率 **70-80%**

⚠️ **Minority Classes**: Accidental & Undetermined 预测困难 (样本量不足)

📊 **Top Features**: `age`, `place_home`, `sex_male` 为最重要预测特征

---

## 📂 Project Structure | 项目结构

```
gun-shot-project/
├── data/
│   ├── raw/Guns_incident_Data.csv          # 原始数据 (62K records)
│   └── processed/                          # 预处理数据 (train/valid/test)
│
├── models/                                 # 训练脚本 & 模型文件
│   ├── 02_xgboost_model.py                # XGBoost训练脚本 ⭐
│   ├── 02_random_forest_model.py
│   ├── 02_smote_decision_tree_model.py
│   ├── 03_evaluate.py                     # 评估脚本
│   └── *.joblib                           # 训练好的模型
│
├── performance/                            # 📊 评估结果 (重要！)
│   ├── model_performance_summary.csv      # 5模型性能对比 (4核心指标)
│   ├── XGBoost_classification_report.csv  # XGBoost详细报告 (per-class)
│   ├── confusion_matrix_xgboost.png       # 混淆矩阵可视化
│   ├── feature_importance_xgboost.png     # 特征重要性排序
│   └── (其他模型的confusion matrix & feature importance)
│
├── notebooks/
│   └── 01_preprocessing.ipynb             # 数据预处理流程
│
├── config.py                              # 配置文件
└── requirements.txt                       # 依赖包
```

---

## 🚀 Quick Start | 快速开始

### 1. Environment Setup | 环境设置

```bash
pip install -r requirements.txt

# Key packages:
# scikit-learn, xgboost, imbalanced-learn, pandas, matplotlib
```

### 2. View Results | 查看结果（无需重新训练）

```bash
# 查看性能汇总
cat performance/model_performance_summary.csv

# 查看可视化结果
open performance/confusion_matrix_xgboost.png
open performance/feature_importance_xgboost.png

# 运行详细评估报告
python models/03_evaluate.py
```

### 3. Reproduce Training | 重现训练（可选）

```bash
# 预处理数据
jupyter notebook notebooks/01_preprocessing.ipynb

# 训练最佳模型
python models/02_xgboost_model.py
```

---

## 📊 For Presentation | Presentation要点

### 1. Problem Statement | 问题定义

**Background**: 枪击事件分类预测，帮助识别事件类型

**Challenge**: 严重类别不平衡 (42:1 ratio)

**Goal**: 开发高性能分类模型，特别关注少数类识别

---

### 2. Methodology | 方法论

**数据处理**:
- 数据清洗 + 特征工程
- 标准化 (StandardScaler) + One-hot编码
- Stratified split (保持类别比例)

**模型策略** (应对类别不平衡):
1. **Class Weighting** - 自动平衡类别权重
2. **SMOTE** - 合成少数类样本
3. **Ensemble Methods** - Random Forest
4. **Gradient Boosting** - XGBoost (最佳) ⭐

**超参数优化**:
- GridSearchCV (72 combinations, 5-fold CV)
- Metric: F1-weighted (更适合不平衡数据)

---

### 3. Results & Analysis | 结果分析

**Overall Performance**:
- Best Model: **XGBoost** (73.9% accuracy, 73.3% F1)
- Improvement: **+16.8%** vs baseline Decision Tree

**Per-Class Performance** (XGBoost - 详见 `XGBoost_classification_report.csv`):
- ✅ **Suicide**: Precision 75%, Recall 83%, F1 79% → **优秀**
- ✅ **Homicide**: Precision 76%, Recall 68%, F1 72% → **良好**
- ⚠️ **Accidental**: Precision 11%, Recall 6%, F1 7% → **困难** (样本仅241)
- ⚠️ **Undetermined**: Precision 2%, Recall 2%, F1 2% → **极困难** (样本仅121)

**Key Insights**:
- 主要类别识别准确，少数类仍具挑战性
- 特征重要性: `age` > `place` > `sex` > `education`
- XGBoost的集成学习优势明显

---

### 4. Visualizations | 可视化说明

**Confusion Matrix** (`confusion_matrix_xgboost.png`):
- 对角线: 预测正确的数量
- 非对角线: 混淆情况
- 反映各类别预测准确性

**Feature Importance** (`feature_importance_xgboost.png`):
- 显示对预测最重要的特征
- 解释模型决策逻辑
- 指导特征选择

**Performance Tables** (CSV格式，可直接查看):
- `model_performance_summary.csv`: 5模型对比 (Accuracy, Recall, F1-Weighted, F1-Macro)
- `XGBoost_classification_report.csv`: XGBoost每个类别的详细指标 (Precision/Recall/F1)
- 便于量化分析和模型选择

---

### 5. Limitations & Future Work | 局限性与改进

**当前局限**:
- ⚠️ 少数类样本不足 (Accidental 240, Undetermined 121)
- ⚠️ 特征维度有限 (26个特征)
- ⚠️ 类别极度不平衡

**改进方向**:
- 📈 收集更多少数类样本
- 🔧 特征工程: 添加交互特征、时间特征
- 🎯 Cost-sensitive learning: 设置不同错误代价
- 🤝 Model ensemble: 结合多个模型
- 🧠 Deep learning: 尝试神经网络方法

---

## 🔗 Key Files Reference | 关键文件索引

### For Presentation Slides:

**数据与方法**:
- Data preprocessing: `notebooks/01_preprocessing.ipynb`
- Best model code: `models/02_xgboost_model.py`
- Evaluation script: `models/03_evaluate.py`

**Results & Figures**:
- 📊 **Models comparison**: `performance/model_performance_summary.csv` (4核心指标)
- 📋 **Detailed report**: `performance/XGBoost_classification_report.csv` (per-class)
- 📈 **Confusion Matrix**: `performance/confusion_matrix_xgboost.png`
- 📉 **Feature Importance**: `performance/feature_importance_xgboost.png`

**Technical Details**:
- Configuration: `config.py`
- Dependencies: `requirements.txt`

---

## 📈 Technical Specifications | 技术规格

### XGBoost Hyperparameters (最佳模型)

```python
XGBClassifier(
    n_estimators=200,          # 200棵树
    max_depth=7,               # 树深度7
    learning_rate=0.1,         # 学习率0.1
    subsample=0.8,             # 80%样本采样
    colsample_bytree=0.8,      # 80%特征采样
    scale_pos_weight='auto',   # 自动处理不平衡
    random_state=42            # 可重复性
)
```

### Evaluation Metrics | 评估指标

Summary表中的4个核心指标 (`model_performance_summary.csv`):
- **Accuracy**: 整体准确率
- **Recall (Weighted)**: 加权召回率
- **F1-Weighted**: 加权F1分数 (主要评估指标) ⭐
- **F1-Macro**: 宏平均F1 (反映少数类表现)

详细报告 (`XGBoost_classification_report.csv`): 每个类别的 Precision/Recall/F1/Support

---

## 👥 Collaboration | 协作说明

**GitHub Repository**: [gun-shot-prediction](https://github.com/yuwu0410/gun-shot-prediction)

**Team Members**:
- Data preprocessing & baseline models
- Model optimization & evaluation
- Documentation & presentation

**Note**: 模型文件(.joblib)不在Git仓库中，通过云盘共享

---

## 📝 Citation & Dataset

**Dataset Source**: Guns Incident Data (62,267 records)

**Features**: Year, Month, Age, Sex, Race, Hispanic, Education, Place, Police Involvement

**Target**: Reason (suicide, homicide, accidental, undetermined)

---

## ⚡ Quick Commands | 常用命令

```bash
# 查看结果
cat performance/model_performance_summary.csv
open performance/  # Mac打开文件夹

# 运行评估
python models/03_evaluate.py

# 重新训练最佳模型
python models/02_xgboost_model.py

# 启动notebook
jupyter notebook notebooks/01_preprocessing.ipynb
```
