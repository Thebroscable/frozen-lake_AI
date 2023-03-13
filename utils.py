import matplotlib.pyplot as plt


def plot_learning(scores, avg_scores, title, filename):
    size = len(scores)
    x = [i+1 for i in range(size)]

    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, scores, color='orange', label='score (orginal)')
    plt.plot(x, avg_scores, color='red', label='score (100 moving averages)')

    plt.title(title)
    plt.legend()

    plt.savefig(filename)






