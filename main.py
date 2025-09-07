import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Creditcard_data.csv")
X, y = df.drop('Class', axis=1), df['Class']
smote = SMOTE(sampling_strategy=0.1, random_state=42)
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_tmp, y_tmp = smote.fit_resample(X, y)
X_bal, y_bal = rus.fit_resample(X_tmp, y_tmp)
balanced = pd.concat([pd.DataFrame(X_bal), pd.Series(y_bal, name='Class')], axis=1)
print("Balanced dataset size: ",len(balanced))


def sample_size(N, Z, p, e):
    n_inf = (Z**2) * p * (1-p) / e**2
    return int(n_inf / (1 + (n_inf-1)/N))

N = len(balanced)
Z = 1.96
p = 0.5
margins = [0.05, 0.04, 0.03, 0.02, 0.01]
sizes = [sample_size(N, Z, p, e) for e in margins]
sizes = [min(s, N) for s in sizes]
print("Adjusted sample sizes: ",sizes)


samples = [
    balanced.sample(n=sizes[i], random_state=42 + i).reset_index(drop=True)
    for i in range(5)
]


samplers = {
    'Sampling1': RandomUnderSampler(random_state=0),
    'Sampling2': RandomOverSampler(random_state=0),
    'Sampling3': SMOTE(random_state=0),
    'Sampling4': SMOTEENN(random_state=0),
    'Sampling5': SMOTETomek(random_state=0)
}

models = {
    'M1': LogisticRegression(max_iter=500, solver='liblinear'),
    'M2': DecisionTreeClassifier(random_state=0),
    'M3': RandomForestClassifier(n_estimators=200, random_state=0),
    'M4': GradientBoostingClassifier(random_state=0),
    'M5': KNeighborsClassifier(n_neighbors=5)
}


results = defaultdict(lambda: defaultdict(float))

for sample in samples:
    X_s, y_s = sample.drop('Class', axis=1), sample['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.3, stratify=y_s, random_state=0
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for s_name, sampler in samplers.items():
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        for m_name, model in models.items():
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            results[m_name][s_name] += accuracy_score(y_test, y_pred)


for m_name in results:
    for s_name in results[m_name]:
        results[m_name][s_name] = round(results[m_name][s_name] / 5, 3)

final_table = pd.DataFrame(results).T
print("\nAccuracy matrix (rows = models, columns = samplers):\n")
print(final_table)

print("\nBest sampler per model:\n")
print(final_table.idxmax(axis=1))
