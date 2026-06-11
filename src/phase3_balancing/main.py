import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier

from src.phase1_baseline.preprocessing import (
    load_data,
    preprocess_data
)

from src.phase1_baseline.utils import (
    evaluate_model
)

from src.phase3_balancing.random_oversampling import (
    apply_random_oversampling
)

from src.phase3_balancing.random_undersampling import (
    apply_random_undersampling
)

from src.phase3_balancing.smote import (
    apply_smote
)

from src.phase3_balancing.adasyn import (
    apply_adasyn
)

from src.phase3_balancing.smoteenn import (
    apply_smoteenn
)

from src.phase3_balancing.smotetomek import (
    apply_smotetomek
)


DATA_PATH = "datasets/tabular/creditcard.csv"


# =========================================================
# TRAIN + EVALUATE
# =========================================================

def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    technique_name
):

    print(f"\n========== {technique_name} ==========")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

    print("\nTraining Model...")

    model.fit(X_train, y_train)

    print("Training Completed!")

    evaluate_model(
        model,
        X_test,
        y_test
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========== PHASE 3 : BALANCING TECHNIQUES ==========")

    # LOAD DATASET
    df = load_data(DATA_PATH)

    # PREPROCESS
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # =====================================================
    # RANDOM OVERSAMPLING
    # =====================================================

    X_ros, y_ros = apply_random_oversampling(
        X_train,
        y_train
    )

    train_and_evaluate(
        X_ros,
        y_ros,
        X_test,
        y_test,
        "RANDOM OVERSAMPLING"
    )

    # =====================================================
    # RANDOM UNDERSAMPLING
    # =====================================================

    X_rus, y_rus = apply_random_undersampling(
        X_train,
        y_train
    )

    train_and_evaluate(
        X_rus,
        y_rus,
        X_test,
        y_test,
        "RANDOM UNDERSAMPLING"
    )

    # =====================================================
    # SMOTE
    # =====================================================

    X_smote, y_smote = apply_smote(
        X_train,
        y_train
    )

    train_and_evaluate(
        X_smote,
        y_smote,
        X_test,
        y_test,
        "SMOTE"
    )

    # =====================================================
    # ADASYN
    # =====================================================

    X_adasyn, y_adasyn = apply_adasyn(
        X_train,
        y_train
    )

    train_and_evaluate(
        X_adasyn,
        y_adasyn,
        X_test,
        y_test,
        "ADASYN"
    )

    # =====================================================
    # SMOTEENN
    # =====================================================

    try:

        X_smoteenn, y_smoteenn = apply_smoteenn(
            X_train,
            y_train
        )

        train_and_evaluate(
            X_smoteenn,
            y_smoteenn,
            X_test,
            y_test,
            "SMOTEENN"
        )

    except Exception as e:

        print("\nSMOTEENN FAILED!")
        print("Error:", e)

    # =====================================================
    # SMOTETOMEK
    # =====================================================

    try:

        X_smotetomek, y_smotetomek = apply_smotetomek(
            X_train,
            y_train
        )

        train_and_evaluate(
            X_smotetomek,
            y_smotetomek,
            X_test,
            y_test,
            "SMOTETOMEK"
        )

    except Exception as e:

        print("\nSMOTETOMEK FAILED!")
        print("Error:", e)

    print("\n========== PHASE 3 COMPLETED ==========")


if __name__ == "__main__":
    main()