import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load features
feature_df = pd.read_csv("feature_dataset.csv")
X = feature_df[["rms_score", "zcr_score", "spectral_score", "vad_score", "whisper_score"]]
y = feature_df["is_noisy"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train neural net
clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# ROC/AUC for neural net
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"MLP (AUC={roc_auc:.2f})", linewidth=2)

# ROC/AUC for individual methods
for method in ["rms_score", "zcr_score", "spectral_score", "vad_score", "whisper_score"]:
    score = X_test[method]
    auc = roc_auc_score(y_test, score)
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.plot(fpr, tpr, label=f"{method} (AUC={auc:.2f})", linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Neural Net vs Individual Methods")
plt.legend()
plt.tight_layout()
plt.savefig("nn_vs_methods_roc_curve.png")
plt.show() 