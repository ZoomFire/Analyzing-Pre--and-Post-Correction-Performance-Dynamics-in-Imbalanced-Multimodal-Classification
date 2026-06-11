from imblearn.over_sampling import ADASYN


def apply_adasyn(X_train, y_train):

    adasyn = ADASYN(random_state=42)

    X_resampled, y_resampled = adasyn.fit_resample(
        X_train,
        y_train
    )

    print("\nADASYN Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled