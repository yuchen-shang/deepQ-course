import matplotlib.pyplot as plt
import numpy as np 

def plot_learning_curve(scores, epsilons):
    N = len(scores)
    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilons, color='C2')
    ax.set_xlabel("training steps")
    ax.set_ylabel('epsilon', color='C2')
    ax.tick_params(axis='y', color='C2')

    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')

    plt.show()