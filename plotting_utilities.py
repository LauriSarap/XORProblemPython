import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_data(dataframe, x_column, y_columns, directory, filename, title, x_label, y_label, is_log_yscale=False, is_log_xscale=True):
    plt.figure(figsize=(10, 6))

    max_value = 0
    min_value = 0
    for column in y_columns:
        if column in y_columns:
            plt.plot(dataframe[x_column], dataframe[column], label=column)
            max_value = max(dataframe[column].max(), dataframe[column].max())
            min_value = min(dataframe[column].min(), dataframe[column].min())
        else:
            print(f"Warning: Column {column} not found in dataframe. Skipping...")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    if is_log_xscale:
        plt.xlabel(f'{x_label} (log scale)')
        plt.xscale('log')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 1000, 10000, 50000]))

    if is_log_yscale:
        plt.yscale('log')
        plt.ylabel(f'{y_label} (log scale)')
        current_ticks = plt.yticks()[0].tolist()
        current_ticks.extend([max_value, min_value])
        plt.yticks(sorted(set(current_ticks)))

    plt.tight_layout()

    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_')]).rstrip()
    plt.savefig(os.path.join(directory, safe_filename))
    plt.close()


def plot_and_save_average(dataframe, directory, column_name, activation_function):
    is_log_yscale = False
    if column_name == 'Loss':
        is_log_yscale = False

    plot_data(
        dataframe=dataframe, x_column='Epoch', y_columns=[column_name],
        directory=directory, filename=f'{column_name}_{activation_function}_average_plot.png',
        title=f'Average {column_name} over epochs for {activation_function} activation function',
        x_label='Epoch', y_label=f'{column_name}', is_log_yscale=is_log_yscale,
    )


def plot_all_gradients_average(dataframe, directory, activation_function):
    gradient_columns = [col for col in dataframe.columns if "GW" in col or "GB" in col]
    plot_data(dataframe=dataframe, x_column='Epoch', y_columns=gradient_columns,
              directory=directory, filename=f'All_average_gradients_{activation_function}_plot.png',
              title=f'All average gradients over epochs for {activation_function} activation function',
              x_label= 'Epoch', y_label='Gradient value')


def plot_all_parameters_average(dataframe, directory, activation_function):
    parameter_columns = [col for col in dataframe.columns if ("W" in col or "B" in col) and ("GW" not in col and "GB" not in col)]
    plot_data(dataframe=dataframe, x_column='Epoch', y_columns=parameter_columns,
              directory=directory, filename=f'All_average_parameters_{activation_function}_plot.png',
              title=f'All average parameters over epochs for {activation_function} activation function',
              x_label='Epoch', y_label='Parameter value')


def plot_and_save_seed(dataframe, directory, column_name, activation_function, seed):
    is_log_yscale = False
    if column_name == 'Loss':
        is_log_yscale = True

    plot_data(
        dataframe=dataframe, x_column='Epoch', y_columns=[column_name],
        directory=directory, filename=f'{column_name}_{activation_function}_seed_{seed}_plot.png',
        title=f'{column_name} over epochs for {activation_function} activation function for network {seed}',
        x_label='Epoch', y_label=f'{column_name}', is_log_yscale=is_log_yscale,
    )


def plot_all_gradients_seed(dataframe, directory, activation_function, seed):
    gradient_columns = [col for col in dataframe.columns if "GW" in col or "GB" in col]
    plot_data(dataframe=dataframe, x_column='Epoch', y_columns=gradient_columns,
              directory=directory, filename=f'All_gradients_{activation_function}_seed_{seed}_plot.png',
              title=f'All gradients over epochs for {activation_function} activation function for network {seed}',
              x_label='Epoch', y_label='Gradient value')


def plot_all_parameters_seed(dataframe, directory, activation_function, seed):
    parameter_columns = [col for col in dataframe.columns if ("W" in col or "B" in col) and ("GW" not in col and "GB" not in col)]
    plot_data(dataframe=dataframe, x_column='Epoch', y_columns=parameter_columns,
              directory=directory, filename=f'All_parameters_{activation_function}_seed_{seed}_plot.png',
              title=f'All parameters over epochs for {activation_function} activation function for network {seed}',
              x_label='Epoch', y_label='Parameter value')


def plot_all_loss_for_one_function(dataframe, directory, activation_function):
    plt.figure(figsize=(10, 6))

    # Sort the columns numerically by seed value for proper plotting order
    sorted_columns = sorted(dataframe.columns, key=lambda x: int(x.split()[-1]) if x not in ['Epoch', 'Seed'] else 0)

    for column in sorted_columns:
        if column not in ['Epoch', 'Seed']:
            plt.plot(dataframe['Epoch'], dataframe[column], label=f'Seed {column}')

    plt.xlabel(f'Epoch (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Loss over epochs for all networks ({activation_function} activation function)')

    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')

    plt.xscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([10, 100, 1000, 10000, 50000]))

    plt.yscale('log')
    upper_ylim = dataframe.drop(['Epoch'], axis=1).stack().quantile(0.9) + 0.2  # 90th percentile + buffer
    plt.ylim(top=upper_ylim)

    plt.tight_layout()

    filename = f'All_seed_losses_{activation_function}_plot.png'
    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_')]).rstrip()
    plt.savefig(os.path.join(directory, safe_filename))
    plt.close()
