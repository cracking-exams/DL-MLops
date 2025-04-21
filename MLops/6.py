import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
y_prob_log = logreg.predict_proba(X_test)[:, 1]

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Create results directory
os.makedirs("results", exist_ok=True)

# Confusion matrix plot function
def plot_cm(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} CM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{name}_cm.png")
    plt.clf()

# Precision-Recall plot function
def plot_pr(y_true, y_prob, name):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(r, p)
    plt.title(f"{name} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"results/{name}_pr.png")
    plt.clf()

# Save visuals
plot_cm(y_test, y_pred_log, "LogReg")
plot_pr(y_test, y_prob_log, "LogReg")
plot_cm(y_test, y_pred_rf, "RandomForest")
plot_pr(y_test, y_prob_rf, "RandomForest")

# Compare performance
ap_log = average_precision_score(y_test, y_prob_log)
ap_rf = average_precision_score(y_test, y_prob_rf)
print("Logistic Regression AP:", round(ap_log, 4))
print("Random Forest AP:", round(ap_rf, 4))
print("Better Model:", "LogReg" if ap_log > ap_rf else "RandomForest")
