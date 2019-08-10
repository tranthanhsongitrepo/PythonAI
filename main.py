import numpy
from sklearn.datasets import load_boston
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

X, y = load_boston(True)
parameters = {'max_depth': 84, 'n_estimators': 25}

regressor = ExtraTreesRegressor(**parameters)

# kfold = KFold(n_splits=8, random_state=6, shuffle=True)

# grid = GridSearchCV(estimator=regressor, param_grid=parameter_grid, cv=kfold)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor.fit(X_train, y_train)

print("Mean absolute error :", round(mean_absolute_error(y_test, regressor.predict(X_test))))
