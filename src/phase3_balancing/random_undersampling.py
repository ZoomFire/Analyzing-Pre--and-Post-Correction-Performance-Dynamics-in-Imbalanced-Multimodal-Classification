from imblearn.under_sampling import RandomUnderSampler


def apply_random_undersampling(X_train, y_train):

    rus = RandomUnderSampler(random_state=42)

    X_resampled, y_resampled = rus.fit_resample(
        X_train,
        y_train
    )

    print("\nRandom Undersampling Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled