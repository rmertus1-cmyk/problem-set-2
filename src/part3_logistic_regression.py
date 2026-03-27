import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression


def run_logistic_regression(df_arrests):

    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    target = 'y'

    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests[target],
        random_state=42
    )

    X_train = df_arrests_train[features]
    y_train = df_arrests_train[target]
    X_test = df_arrests_test[features]

    param_grid = {'C': [0.01, 0.1, 1.0]}

    lr_model = LogisticRegression(max_iter=1000)

    gs_cv = GridSearchCV(
        lr_model,
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    gs_cv.fit(X_train, y_train)

    best_c = gs_cv.best_params_['C']

    print("Best C:", best_c)

    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_lr'] = gs_cv.predict(X_test)
    df_arrests_test['pred_prob_lr'] = gs_cv.predict_proba(X_test)[:, 1]

    return df_arrests_train, df_arrests_test, gs_cv