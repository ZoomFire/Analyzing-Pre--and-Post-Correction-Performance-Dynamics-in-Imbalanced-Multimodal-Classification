from imblearn.combine import SMOTEENN


def apply_smoteenn(X_train, y_train):

    smoteenn = SMOTEENN(random_state=42)

    X_resampled, y_resampled = smoteenn.fit_resample(
        X_train,
        y_train
    )

    print("\nSMOTEENN Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled