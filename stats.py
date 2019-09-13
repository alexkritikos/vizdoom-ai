from matplotlib import pyplot as plt
import numpy as np


def read_stats(file):
    mavg_score = []
    mavg_ammo_left = []
    mavg_kill_counts = []

    with open(file, 'r') as fp:
        for cnt, line in enumerate(fp):
            if cnt == 1:
                max_score = int(line[11:])
            elif cnt == 2:
                mavg_score = line[13:-2]
                mavg_score = [x.strip() for x in mavg_score.split(',')]
            elif cnt == 4:
                mavg_ammo_left = line[17:-2]
                mavg_ammo_left = [x.strip() for x in mavg_ammo_left.split(',')]
            elif cnt == 5:
                mavg_kill_counts = line[19:-2]
                mavg_kill_counts = [x.strip() for x in mavg_kill_counts.split(',')]

        for i in range(0, len(mavg_score)):
            mavg_score[i] = float(mavg_score[i])
        for i in range(0, len(mavg_ammo_left)):
            mavg_ammo_left[i] = float(mavg_ammo_left[i])
        for i in range(0, len(mavg_kill_counts)):
            mavg_kill_counts[i] = float(mavg_kill_counts[i])
        return max_score, mavg_score, mavg_ammo_left, mavg_kill_counts


def generate_max_score_chart(ddqn, c51):
    algorithms = ('DDQN', 'C51')
    y_pos = np.arange(len(algorithms))
    performance = [ddqn, c51]

    plt.bar(y_pos, performance, width=0.4, align='center', alpha=0.8)
    plt.xticks(y_pos, algorithms)
    plt.xlabel('Αλγόριθμος')
    plt.ylabel('Σκορ')
    plt.title('Μέγιστο Σκορ Αλγορίθμων')

    plt.show()


def generate_plot(ddqn_list, c51_list, x_label, y_label, title):
    timesteps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
                 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
    plt.plot(timesteps, ddqn_list, label='DDQN')
    plt.plot(timesteps, c51_list, label='C51')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    ddqn_max_score, ddqn_mavg_score, ddqn_mavg_ammo_left, ddqn_mavg_kill_counts = \
        read_stats("statistics/ddqn/ddqn_stats.txt")
    c51_max_score, c51_mavg_score, c51_mavg_ammo_left, c51_mavg_kill_counts = \
        read_stats("statistics/c51/c51_ddqn_stats.txt")
    c51_mavg_score, c51_mavg_ammo_left, c51_mavg_kill_counts = c51_mavg_score[:-5], c51_mavg_ammo_left[:-5], \
                                                               c51_mavg_kill_counts[:-5]
    generate_max_score_chart(ddqn_max_score, c51_max_score)
    generate_plot(ddqn_mavg_score, c51_mavg_score, 'Βήματα', 'Σκορ', 'Μέσο Σκορ Αλγορίθμων')
    generate_plot(ddqn_mavg_ammo_left, c51_mavg_ammo_left, 'Βήματα', 'Σφαίρες', 'Μέσες Υπολειπόμενες Σφαίρες')
    generate_plot(ddqn_mavg_kill_counts, c51_mavg_kill_counts, 'Βήματα', 'Εξοντώσεις', 'Μέσες Εξοντώσεις')
