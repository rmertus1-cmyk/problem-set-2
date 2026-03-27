from part2_preprocessing import preprocess_data
from part3_logistic_regression import run_logistic_regression
from part4_decision_tree import run_decision_tree
from part5_calibration_plot import run_calibration


def main():
    df_arrests = preprocess_data()

    df_arrests_train, df_arrests_test, _ = run_logistic_regression(df_arrests)

    df_arrests_train, df_arrests_test, _ = run_decision_tree(
        df_arrests_train, df_arrests_test
    )

    run_calibration(df_arrests_test)


if __name__ == "__main__":
    main()