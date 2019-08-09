from sklearn.datasets import load_boston
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X, y = load_boston(True)
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

regressor = ExtraTreesRegressor(**params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor.fit(X_train, y_train)

print("Mean absolute error :", round(mean_absolute_error(y_test, regressor.predict(X_test))))