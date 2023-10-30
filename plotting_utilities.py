import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_and_save_average(dataframe, directory, column_name, activation_function):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['Epoch'], dataframe[column_name], label=column_name, color='blue')
    plt.xlabel('Epoch (log scale)')
    plt.ylabel(f'{column_name} (log scale)')
    plt.title(f'Average {column_name} over epochs for {activation_function} activation function')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    if column_name == 'Loss':
        max_value = dataframe[column_name].max()
        min_value = dataframe[column_name].min()
        current_ticks = plt.yticks()[0].tolist()
        current_ticks.extend([max_value, min_value])
        plt.yticks(sorted(set(current_ticks)))
        #plt.yticks([0, 0.01, 0.1, 0.5], [0, 0.01, 0.1, 0.5])  # Set y-ticks
        #plt.ylim(bottom=0)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{column_name}_{activation_function}_plot.png'))
    plt.close()


def plot_and_save_seed(dataframe, directory, column_name, activation_function, seed):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['Epoch'], dataframe[column_name], label=column_name, color='blue')
    plt.xlabel('Epoch (log scale)')
    plt.ylabel(column_name)
    plt.title(f'{column_name} over epochs for {activation_function} activation function for network {seed}')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    if column_name == 'Loss':
        plt.yscale('log')
        max_value = dataframe[column_name].max()
        min_value = dataframe[column_name].min()
        current_ticks = plt.yticks()[0].tolist()
        current_ticks.extend([max_value, min_value])
        plt.yticks(sorted(set(current_ticks)))
        #plt.yticks([0, 0.01, 0.1, 0.5], [0, 0.01, 0.1, 0.5])  # Set y-ticks
        #plt.ylim(bottom=0)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{column_name}_{activation_function}_seed_{seed}_plot.png'))
    plt.close()


def plot_all_gradients_average(dataframe, directory, activation_function):
    plt.figure(figsize=(10, 6))

    gradient_columns = [col for col in dataframe.columns if "GW" in col or "GB" in col]

    for column in gradient_columns:
        plt.plot(dataframe['Epoch'], dataframe[column], label=column)

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Gradient value')
    plt.title(f'All average gradients over epochs for {activation_function} activation function')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'All_average_gradients_{activation_function}_plot.png'))
    plt.close()


def plot_all_parameters_average(dataframe, directory, activation_function):
    plt.figure(figsize=(10, 6))

    parameter_columns = [col for col in dataframe.columns if
                        ("W" in col or "B" in col) and ("GW" not in col and "GB" not in col)]

    for column in parameter_columns:
        plt.plot(dataframe['Epoch'], dataframe[column], label=column)

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Parameter value')
    plt.title(f'All average parameters over epochs for {activation_function} activation function')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'All_average_parameters_{activation_function}_plot.png'))
    plt.close()


def plot_all_gradients_seed(dataframe, directory, activation_function, seed):
    plt.figure(figsize=(10, 6))

    gradient_columns = [col for col in dataframe.columns if "GW" in col or "GB" in col]

    for column in gradient_columns:
        plt.plot(dataframe['Epoch'], dataframe[column], label=column)

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Gradient value')
    plt.title(f'All gradients over epochs for {activation_function} activation function for network {seed}')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'All_gradients_{activation_function}_seed_{seed}_plot.png'))
    plt.close()


def plot_all_parameters_seed(dataframe, directory, activation_function, seed):
    plt.figure(figsize=(10, 6))

    parameter_columns = [col for col in dataframe.columns if
                        ("W" in col or "B" in col) and ("GW" not in col and "GB" not in col)]

    for column in parameter_columns:
        plt.plot(dataframe['Epoch'], dataframe[column], label=column)

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('Parameter value')
    plt.title(f'All parameters over epochs for {activation_function} activation function for network {seed}')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 500, 1000, 10000, 50000]))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'All_parameters_{activation_function}_seed_{seed}_plot.png'))
    plt.close()
