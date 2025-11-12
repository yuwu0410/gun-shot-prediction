# Model Comparison: Decision Trees vs Logistic Regression

**Performance**: LogReg (78.2% accuracy) outperforms DT/XGBoost (73.9%) by 4.3%. Key differences explaining this gap:

(1) **Problem framing** 
DT keeps 4 classes (42:1 imbalance, Undetermined only 2% recall). LogReg merges Accidental+Undetermined into "Others" (26:1 imbalance, Others 14% recall) => simplification helps but can't distinguish the two types anymore.

(2) **Architecture** 
DT uses single multiclass XGBoost, while LogReg uses OvA ensemble (3 binary specialists) => binary decision boundaries clearer than multiclass for extreme imbalance, each model focuses independently.

(3) **Major class performance** - Both succeed: Suicide/Homicide 70-80%. LogReg has higher Suicide precision (87% vs 75%), DT has higher Suicide recall (83% vs 81%).

(4) **Minority class trade-off** - DT preserves original 4 classes but fails catastrophically (2-11% metrics). LogReg's merged approach achieves 14% recall (7x better than DT's Undetermined) but can't distinguish Accidental from Undetermined.

(5) **Root cause & why LogReg wins** 
Both face overlapping demographic features (same age/sex/race/place profile across classes) + sample scarcity. But LogReg handles it better: 
(a), Merging gives 474 combined samples vs 121+241 separate, more data per class. 
(b), Binary specialists (OvA) naturally suit extreme imbalance - each model only needs to separate one class from rest, simpler than DT's 4-way multiclass decision. 
(c), Fewer inter-class boundaries: 3 classes need fewer decision boundaries than 4. DT's 121-241 minority samples spread across 4 classes (small number of data per class) while LogReg concentrates 474 samples into 3 classes. 
BUT, Both still fail on minority class data due to feature overlap + data scarcity. 

