# ⚽ Scoutium Player Potential Prediction

This repository contains a machine learning project that predicts football players' potential classes based on scout evaluations from the **Scoutium** platform.

The goal is to classify players into:
- `average`
- `highlighted`

using their match-based attribute scores given by professional scouts.

---

## 🚀 Project Overview

Scouts (talent evaluators) rate players on various technical, physical and tactical attributes during matches.  
In addition to these attribute scores, each player receives a final **potential label** such as `average` or `highlighted`.

In this project we:

- Merge attribute-level data with potential labels
- Filter specific positions and labels
- Transform the data into a **player–attribute matrix** using pivot tables
- Encode categorical targets and scale numerical features
- Train multiple classification models
- Perform hyperparameter optimization
- Analyze feature importance

---

## 📂 Dataset

The dataset comes from the Scoutium platform and consists of two main CSV files:

### 1️⃣ `scoutium_attributes.csv`

Contains attribute evaluations given to players during matches.

| Column            | Description                                               |
|-------------------|-----------------------------------------------------------|
| `task_response_id`| Set of evaluations by a scout for a specific match       |
| `match_id`        | Match identifier                                          |
| `evaluator_id`    | Scout (evaluator) identifier                              |
| `player_id`       | Player identifier                                         |
| `position_id`     | Player's position in that match                           |
| `analysis_id`     | Set of attribute ratings for a player                     |
| `attribute_id`    | Evaluated attribute identifier                            |
| `attribute_value` | Score given to the attribute                              |

#### Position ID Reference

1: Goalkeeper  
2: Centre-back  
3: Right-back  
4: Left-back  
5: Defensive midfielder  
6: Central midfielder  
7: Right winger  
8: Left winger  
9: Attacking midfielder  
10: Striker  

---

### 2️⃣ `scoutium_potential_labels.csv`

Contains the **target variable**: the final potential decision made by the scout.

| Column            | Description                                  |
|-------------------|----------------------------------------------|
| `task_response_id`| Evaluation set ID                            |
| `match_id`        | Match identifier                             |
| `evaluator_id`    | Scout identifier                             |
| `player_id`       | Player identifier                            |
| `potential_label` | Final potential label (`average` / `highlighted`) |

---

## 🧠 Modeling Approach

Main steps in the notebook:

1. **Data Merging & Cleaning**
   - Merge attributes with potential labels using keys:
     - `["task_response_id", "match_id", "evaluator_id", "player_id"]`
   - Filter out undesired positions (e.g. goalkeepers) if needed
   - Drop samples with specific labels (e.g. `below_average`)

2. **Feature Engineering**
   - Create a pivot table to obtain a **player × attribute** matrix:
     ```python
     df_pivot = df.pivot_table(
         index=["player_id", "position_id", "potential_label"],
         columns=["attribute_id"],
         values="attribute_value"
     )
     ```
   - Encode `potential_label` with `LabelEncoder`
   - Scale numerical features using `StandardScaler`

3. **Model Training & Hyperparameter Optimization**
   - Train multiple classifiers:
     - RandomForestClassifier  
     - GradientBoostingClassifier  
     - CatBoostClassifier  
     - LGBMClassifier  
   - Use `GridSearchCV` and `cross_validate` for:
     - Accuracy  
     - F1-score  
     - ROC–AUC  
     - Precision  
     - Recall  

4. **Feature Importance**
   - Visualize feature importances of the best model using a custom `plot_importance` function.

---

## 📦 Requirements

Main Python dependencies (not exhaustive):

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
catboost
lightgbm

## 📦 Installation

Install all required dependencies with:

```bash
pip install -r requirements.txt

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
---

### 2️⃣ Install the dependencies

```bash
pip install -r requirements.txt
---

### 3️⃣ Place the dataset files

Make sure the dataset CSV files are located in the correct folder (e.g., `data/`).

---

### 4️⃣ Run the notebook

```bash
jupyter notebook


Then open the main notebook file (e.g. `scoutium_player_potential.ipynb`).

---

## 📊 Results (Summary)

- **Best model:** LightGBM  
- **CV Accuracy:** `88.94%`  
- **CV F1-score:** `0.6782`  
- **ROC–AUC:** `0.8763`  

---

## ✨ Notes

This project is ideal for:

- Practicing **classification** on real-world sports data  
- Working with **categorical labels**, **pivot tables**, and **feature scaling**  
- Experimenting with **tree-based ensemble models**, such as:
  - Random Forest  
  - Gradient Boosting  
  - CatBoost  
  - LightGBM  


