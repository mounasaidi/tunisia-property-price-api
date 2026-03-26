import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df):
    plt.hist(df['prix_dt'], bins=50)
    plt.title("Distribution des prix")
    plt.show()


def plot_correlation(df, cols):
    sns.heatmap(df[cols].corr(), annot=True)
    plt.title("Corrélation")
    plt.show()