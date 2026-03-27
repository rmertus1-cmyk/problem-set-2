'''
PART 5: Calibration-light
'''

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


# already given
def calibration_plot(y_true, y_prob, n_bins=10):
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)

    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()


def run_calibration(df_arrests_test):
    y_true = df_arrests_test['y']

    # Logistic Regression
    print("Logistic Regression Calibration Plot")
    calibration_plot(y_true, df_arrests_test['pred_prob_lr'], n_bins=5)

    # Decision Tree
    print("Decision Tree Calibration Plot")
    calibration_plot(y_true, df_arrests_test['pred_prob_dt'], n_bins=5)

    print("Which model is more calibrated?")
    print("Logistic regression is typically more calibrated because it produces smoother probability estimates, while decision trees tend to be more overconfident.")


    # PPV top 50
    top50_lr = df_arrests_test.sort_values('pred_prob_lr', ascending=False).head(50)
    top50_dt = df_arrests_test.sort_values('pred_prob_dt', ascending=False).head(50)

    ppv_lr = top50_lr['y'].mean()
    ppv_dt = top50_dt['y'].mean()

    print("PPV (Top 50) Logistic Regression:", ppv_lr)
    print("PPV (Top 50) Decision Tree:", ppv_dt)

    # AUC
    auc_lr = roc_auc_score(y_true, df_arrests_test['pred_prob_lr'])
    auc_dt = roc_auc_score(y_true, df_arrests_test['pred_prob_dt'])

    print("AUC Logistic Regression:", auc_lr)
    print("AUC Decision Tree:", auc_dt)

    print("Do both metrics agree that one model is more accurate than the other?")
    if (ppv_lr > ppv_dt and auc_lr > auc_dt):
        print("Yes, logistic regression is better by both metrics.")
    elif (ppv_dt > ppv_lr and auc_dt > auc_lr):
        print("Yes, decision tree is better by both metrics.")
    else:
        print("No, the metrics do not fully agree.")