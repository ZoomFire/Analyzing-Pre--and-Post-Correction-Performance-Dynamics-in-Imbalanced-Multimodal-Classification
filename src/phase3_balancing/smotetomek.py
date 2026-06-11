from imblearn.combine import SMOTETomek


def apply_smotetomek(X_train, y_train):

    smotetomek = SMOTETomek(random_state=42)

    X_resampled, y_resampled = smotetomek.fit_resample(
        X_train,
        y_train
    )

    print("\nSMOTETomek Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled