from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nSMOTE Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled