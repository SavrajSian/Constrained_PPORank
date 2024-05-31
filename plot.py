import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


def parse_log_file(file_path):
    epochs = []
    train_ndcgs = []
    test_ndcgs = []
    with open(file_path, 'r') as file:
        for line in file:
            if "train_ndcg" in line:
                parts = line.split('-')
                # Extracting epoch number
                epoch = int(parts[5].split('@')[1].split(':')[0])
                # Extracting train_ndcg
                train_ndcg = float(parts[5].split('train_ndcg')[1].split(',')[0].strip())
                # Extracting test_ndcg
                test_ndcg = float(parts[5].split('test_ndcg')[1].split(',')[0].strip())

                epochs.append(epoch)
                train_ndcgs.append(train_ndcg)
                test_ndcgs.append(test_ndcg)
    return epochs, train_ndcgs, test_ndcgs


# File path to your log file

# Parse the log file

# Define the custom color palette

# Update rcParams to use the custom color palette
#plt.rcParams['axes.prop_cycle'] = cycler(color=custom_palette)


def pretty_plot_ndcg(file_path, title, save_name):

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.style.use('fivethirtyeight')

    epochs, train_ndcgs, test_ndcgs = parse_log_file(file_path)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_ndcgs, label='Train NDCG', linewidth=2.5)
    plt.plot(epochs, test_ndcgs, label='Test NDCG', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graphs/' + save_name + '.png', dpi=300)
    plt.show()



file_path = 'plot_files/PPORank_FYP_lr1.7e-4_15xlrsched_epochs700_26thmay.e77504'
title = 'Catastrophic Forgetting'
save_name = 'catastrophic_forgetting'

pretty_plot_ndcg(file_path, title, save_name)

