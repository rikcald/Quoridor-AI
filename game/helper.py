import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.ylim(ymin=0)
