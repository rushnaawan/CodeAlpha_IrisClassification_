import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)

# load data
df = pd.read_csv("Iris.csv").drop(columns="Id")
df["Species"] = df["Species"].str.replace("Iris-", "")
features = list(df.columns[:4])
classes = sorted(df["Species"].unique())
print(df.shape)
print(df.isnull().sum())
print(df["Species"].value_counts())
# a few rows repeat in iris, they are real flowers so i am keeping them
print(df.duplicated().sum(), "duplicate rows")
print(df.describe())

# quick look at the data
sns.pairplot(df, hue="Species")
plt.savefig("outputs/pairplot.png")
plt.close()

X = df[features]
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

rows = []
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=5).mean()
    m.fit(X_train, y_train)
    test = accuracy_score(y_test, m.predict(X_test))
    rows.append((name, round(cv, 3), round(test, 3)))
    print(name, "cv:", round(cv, 3), "test:", round(test, 3))

pd.DataFrame(rows, columns=["model", "cv_accuracy", "test_accuracy"]).to_csv("outputs/results.csv", index=False)

# pick the best one using cv score (not the test score)
# knn and svm have the same cv score, max() takes the first one
best_name = max(rows, key=lambda r: r[1])[0]
best = models[best_name]
pred = best.predict(X_test)
print("\nbest model:", best_name)
print(classification_report(y_test, pred, target_names=classes))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=classes), display_labels=classes).plot()
plt.title(best_name)
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# try one new flower
new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=features)
print("prediction for new flower:", best.predict(new)[0])
