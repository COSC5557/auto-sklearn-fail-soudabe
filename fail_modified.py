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
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))


automl = AutoSklearnClassifier(time_left_for_this_task=300, per_run_time_limit=60, resampling_strategy='cv')
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))


# RF Accuracy 0.6525
# AutoML Accuracy 0.6625
