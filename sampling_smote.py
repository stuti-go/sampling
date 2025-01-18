import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt

data_url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(data_url)

X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

balanced_data = pd.concat([pd.DataFrame(X_balanced), pd.DataFrame(y_balanced, columns=['Class'])], axis=1)

def simple_random_sampling(df):
    return df.sample(frac=1, random_state=42)

def stratified_sampling(df):
    return df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))

def systematic_sampling(df):
    return df.iloc[::5, :]

def cluster_sampling(df):
    return df.sample(frac=0.5, random_state=42)

def reservoir_sampling(df, sample_size):
    return df.sample(n=sample_size, random_state=42)

z = 1.96
p = 0.5
e = 0.05

n = int((z**2 * p * (1 - p)) / (e**2))
sampling_methods = [
    simple_random_sampling,
    stratified_sampling,
    systematic_sampling,
    cluster_sampling,
    lambda df: reservoir_sampling(df, sample_size=n)
]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = []

for i, sampling_method in enumerate(sampling_methods, start=1):
    sample = sampling_method(balanced_data)
    X_sample = sample.drop('Class', axis=1)
    y_sample = sample['Class']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_sample, y_sample, cv=skf, scoring='f1')
        f1_score = np.mean(cv_scores)
        results.append({
            "Sampling Technique": f"Sampling{i}",
            "Model": model_name,
            "F1 Score": f1_score
        })

results_df = pd.DataFrame(results)
results_df.to_csv("sampling_results_f1.csv", index=False)

results_pivot = results_df.pivot(index="Model", columns="Sampling Technique", values="F1 Score")

print("F1 Score Results Table:")
print(results_pivot)

plt.figure(figsize=(10, 6))
sns.heatmap(results_pivot, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("F1 Score of Models with Different Sampling Techniques")
plt.ylabel("Model")
plt.xlabel("Sampling Technique")
plt.show()