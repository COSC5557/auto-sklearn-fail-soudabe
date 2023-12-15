import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from autosklearn.classification import AutoSklearnClassifier

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(X)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat_train_rf = clf.predict(X_train)
y_hat_test_rf = clf.predict(X_test)
rf_train_acc = sklearn.metrics.accuracy_score(y_train, y_hat_train_rf)
rf_test_acc = sklearn.metrics.accuracy_score(y_test, y_hat_test_rf)
print("RF Train Accuracy", rf_train_acc)
print("RF Test Accuracy", rf_test_acc)

automl = AutoSklearnClassifier(time_left_for_this_task=300)
automl.fit(X_train, y_train)
y_hat_train_automl = automl.predict(X_train)
y_hat_test_automl = automl.predict(X_test)
automl_train_acc = sklearn.metrics.accuracy_score(y_train, y_hat_train_automl)
automl_test_acc = sklearn.metrics.accuracy_score(y_test, y_hat_test_automl)
print("AutoML Train Accuracy", automl_train_acc)
print("AutoML Test Accuracy", automl_test_acc)
