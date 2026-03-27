'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth.
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`.
- Initialize the GridSearchCV using the decision tree model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`.
- Run the model
- What was the optimal value for max_depth? Did it have the most or least regularization? Or in the middle?
- Now predict for the test set. Name this column `pred_dt`
- Save dataframe(s) as .csv('s) in `data/`
'''

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def run_decision_tree(df_arrests_train, df_arrests_test):
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    target = 'y'

    X_train = df_arrests_train[features]
    y_train = df_arrests_train[target]
    X_test = df_arrests_test[features]

    param_grid_dt = {
        'max_depth': [1, 3, 5]
    }

    dt_model = DecisionTreeClassifier(random_state=42)

    gs_cv_dt = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid_dt,
        cv=5,
        scoring='accuracy'
    )

    gs_cv_dt.fit(X_train, y_train)

    best_depth = gs_cv_dt.best_params_['max_depth']

    print("What was the optimal value for max_depth?")
    print(best_depth)

    print("Did it have the most or least regularization? Or in the middle?")
    if best_depth == min(param_grid_dt['max_depth']):
        print("It had the most regularization.")
    elif best_depth == max(param_grid_dt['max_depth']):
        print("It had the least regularization.")
    else:
        print("It was in the middle.")

    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)
    df_arrests_test['pred_prob_dt'] = gs_cv_dt.predict_proba(X_test)[:, 1]

    df_arrests_train.to_csv('src/data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('src/data/df_arrests_test.csv', index=False)

    return df_arrests_train, df_arrests_test, gs_cv_dt