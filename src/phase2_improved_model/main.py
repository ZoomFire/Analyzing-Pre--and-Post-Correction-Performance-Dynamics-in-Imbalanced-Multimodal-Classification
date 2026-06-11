from src.phase1_baseline.preprocessing import (
    load_data,
    preprocess_data
)

from src.phase1_baseline.utils import (
    evaluate_model
)

from src.phase2_improved_model.feature_engineering import (
    feature_engineering
)

from src.phase2_improved_model.improved_model import (
    train_improved_model
)

from src.phase2_improved_model.xgboost_model import (
    train_xgboost_model
)

from src.phase2_improved_model.lightgbm_model import (
    train_lightgbm_model
)


DATA_PATH = "datasets/tabular/creditcard.csv"


def main():

    print("\n========== PHASE 2 : IMPROVED MODELS ==========")

    # LOAD DATASET
    df = load_data(DATA_PATH)

    # FEATURE ENGINEERING
    df = feature_engineering(df)

    # PREPROCESS DATA
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    print("\n========== RANDOM FOREST ==========")

    rf_model = train_improved_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # =====================================================
    # XGBOOST
    # =====================================================

    print("\n========== XGBOOST ==========")

    xgb_model = train_xgboost_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    # =====================================================
    # LIGHTGBM
    # =====================================================

    print("\n========== LIGHTGBM ==========")

    lgbm_model = train_lightgbm_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_model(
        lgbm_model,
        X_test,
        y_test
    )

    print("\n========== PHASE 2 COMPLETED ==========")


if __name__ == "__main__":
    main()