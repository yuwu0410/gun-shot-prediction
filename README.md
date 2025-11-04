# Gun Shot Incident Prediction

## Quick Start

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 添加数据集
将 `Guns_incident_Data.csv` 放到 `data/raw/` 目录

### 3. 运行预处理
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```
运行所有cells后，会在 `data/processed/` 生成：
- X_train.parquet, X_valid.parquet, X_test.parquet
- y_train.csv, y_valid.csv, y_test.csv
- preprocessor.joblib
- feature_names.csv
- data_dictionary.json

### 4. 训练决策树模型（带超参数优化）
```bash
python models/02_train_decision_tree.py
```
**特性**：
- 使用GridSearchCV自动搜索最佳超参数
- 72种参数组合，5折交叉验证
- 针对类别不平衡问题优化（使用F1-weighted评分）
- 预计运行时间：5-10分钟

训练完成后会在 `models/` 目录生成：
- decision_tree_model.joblib

训练脚本会输出：
- 最佳超参数组合
- 交叉验证F1分数
- 前5名参数配置对比
- 训练集和验证集的准确率
- 验证集的详细分类报告（关注少数类表现）

### 5. 模型评估与优化（TODO）

**当前状态**：
- ✅ 初始模型已训练（Decision Tree with Grid Search）
- ✅ Validation F1 = 0.62（少数类表现差：accidental F1=0.07, undetermined F1=0.04）

**TODO - 阶段1：模型开发（使用Validation Set）**

1. 尝试改进模型（在 train set 上训练，在 validation set 上评估）：
   - 选项A：Random Forest（更鲁棒）
   - 选项B：SMOTE过采样 + 决策树
   - 选项C：调整class_weight权重
   - 选项D：XGBoost/LightGBM

2. 比较不同模型在 validation set 上的表现

3. 选出最佳模型

**TODO - 阶段2：最终评估（使用Test Set，只运行1次）**

1. 完成 `models/03_evaluate.py`：
   - 加载最佳模型和test数据
   - 计算评估指标（accuracy, precision, recall, F1）
   - 生成混淆矩阵可视化
   - 分析特征重要性
   - 各类别性能对比

2. 运行最终评估：
   ```bash
   python models/03_evaluate.py
   ```

### 6. 项目结构
```
gun-shot-project/
├── data/
│   ├── raw/              # 原始CSV文件
│   └── processed/        # 预处理后的数据
├── notebooks/
│   └── 01_preprocessing.ipynb         # 数据预处理notebook
├── models/
│   ├── 02_train_decision_tree.py      # 决策树训练脚本（GridSearchCV）
│   ├── 03_evaluate.py                 # 模型评估脚本（待完成）
│   └── decision_tree_model.joblib     # 训练好的模型
├── config.py            # 路径配置
└── requirements.txt     # 项目依赖
```