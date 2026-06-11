from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_model(X_train, y_train):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)

    return grid.best_estimator_