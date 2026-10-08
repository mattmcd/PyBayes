# %%
from startup import np, pd, plt, sns

# %%
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
penguins = sns.load_dataset("penguins")
X = penguins.drop(columns="species")
y = penguins["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
# INFO: You are downloading 'tabicl-classifier-v2-20260212.ckpt', the latest best-performing version, used in our TabICLv2 paper
clf = TabICLClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
clas_report = classification_report(y_test, y_pred)
print(clas_report)