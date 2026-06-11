import matplotlib.pyplot as plt


def plot_class_distribution(df):

    toxic_counts = df.iloc[:, 2:].sum()

    plt.figure(figsize=(12, 5))

    toxic_counts.plot(kind="bar")

    plt.title(
        "Toxic Comment Class Distribution"
    )

    plt.xlabel("Classes")

    plt.ylabel("Samples")

    plt.tight_layout()

    plt.show()