from imblearn.over_sampling import RandomOverSampler


def apply_random_oversampling(X_train, y_train):

    ros = RandomOverSampler(random_state=42)

    X_resampled, y_resampled = ros.fit_resample(
        X_train,
        y_train
    )

    print("\nRandom Oversampling Applied!")
    print("Before Balancing:", y_train.value_counts())
    print("After Balancing:\n", y_resampled.value_counts())

    return X_resampled, y_resampled