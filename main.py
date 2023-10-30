import train_model
import plotting_utilities
import pandas as pd
import os


# Settings
STARTING_SEED = 1
PARAMETER_SAVING_FREQUENCY = 100
NUM_RUNS = 10
activation_functions = ['tanh', 'sigmoid', 'relu']


def run_for_activation(activation_function):
    # Run the model and save the raw data
    print(f"Running for activation function: {activation_function}")
    directory = f'./{activation_function}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    raw_data_file = os.path.join(directory, f'{activation_function}_raw_data.csv')

    if not os.path.exists(raw_data_file):
        all_data = []
        columns = ["Seed & Epoch", "Loss", "L1 W1,1", "L1 W2,1", "L1 W1,2", "L1 W2,2", "L1 B1", "L1 B2", "L2 W1,1",
                   "L2 W2,1", "L2 B1", "L1 GW1,1", "L1 GW2,1", "L1 GW1,2", "L1 GW2,2", "L1 GB1", "L1 GB2", "L2 GW1,1",
                   "L2 GW2,1", "L2 GB1", "Predicted Y1", "Predicted Y2", "Predicted Y3", "Predicted Y4"]

        for run in range(NUM_RUNS):
            SEED = STARTING_SEED + run
            print(f"Running with seed: {SEED}")
            data = train_model.train(SEED=SEED, ACTIVATION_FUNCTION=activation_function, PARAMETER_SAVING_FREQUENCY=PARAMETER_SAVING_FREQUENCY)

            for i in range(len(data['losses'])):
                epoch = i * PARAMETER_SAVING_FREQUENCY
                predicted_outputs = data['predicted_outputs'][i].flatten()
                row = ([f"Seed {SEED} / Epoch {epoch}", data['losses'][i]] + list(data['l1_weights_epoch'][i].flatten()) + \
                      list(data['l1_biases_epoch'][i].flatten()) + list(data['l2_weights_epoch'][i].flatten()) + \
                      list(data['l2_biases_epoch'][i].flatten()) + list(data['l1_weight_gradients_epoch'][i].flatten()) + \
                      list(data['l1_bias_gradients_epoch'][i].flatten()) + list(data['l2_weight_gradients_epoch'][i].flatten()) + \
                      list(data['l2_bias_gradients_epoch'][i].flatten()) + list(predicted_outputs))
                all_data.append(row)

        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(raw_data_file, index=False)

    # Plot the data for each seed
    raw_data = pd.read_csv(raw_data_file)
    raw_data['Epoch'] = raw_data['Seed & Epoch'].apply(lambda x: int(x.split(' ')[-1]))
    for run in range(NUM_RUNS):
        SEED = STARTING_SEED + run
        seed_data = raw_data[raw_data['Seed & Epoch'].str.startswith(f"Seed {SEED} /")]
        seed_specific_directory = os.path.join(directory, f'seed_{SEED}')
        if not os.path.exists(seed_specific_directory):
            print(f'Plotting parameters for seed {SEED}')
            os.makedirs(seed_specific_directory)
            for column in seed_data.columns:
                if column not in ['Seed & Epoch', 'Epoch']:
                    plotting_utilities.plot_and_save_seed(seed_data, seed_specific_directory, column, activation_function, SEED)

        if os.path.exists(seed_specific_directory):
            seed_multiple_plots = os.path.join(seed_specific_directory, 'all_plots')
            if not os.path.exists(seed_multiple_plots):
                os.makedirs(seed_multiple_plots)
                plotting_utilities.plot_all_gradients_seed(seed_data, seed_multiple_plots, activation_function, SEED)
                plotting_utilities.plot_all_parameters_seed(seed_data, seed_multiple_plots, activation_function, SEED)

    # Average the data
    averaged_data_file = os.path.join(directory, f'{activation_function}_averaged_data.csv')
    if not os.path.exists(averaged_data_file):
        #raw_data = pd.read_csv(raw_data_file)
        #raw_data['Epoch'] = raw_data['Seed & Epoch'].apply(lambda x: int(x.split(' ')[-1]))
        average_losses = raw_data.groupby('Epoch').mean().reset_index()
        average_losses.to_csv(averaged_data_file, index=False)

    averaged_data_plots_directory = os.path.join(directory, 'average')
    averaged_data = pd.read_csv(averaged_data_file)
    if not os.path.exists(averaged_data_plots_directory):
        print(f'Plotting parameters for averaged data')
        os.makedirs(averaged_data_plots_directory)
        for column in averaged_data.columns:
            if column != 'Epoch':
                plotting_utilities.plot_and_save_average(averaged_data, averaged_data_plots_directory, column, activation_function)

    if os.path.exists(averaged_data_plots_directory):
        average_multiple_plots = os.path.join(averaged_data_plots_directory, 'all_plots')
        if not os.path.exists(average_multiple_plots):
            os.makedirs(average_multiple_plots)
            plotting_utilities.plot_all_gradients_average(averaged_data, average_multiple_plots, activation_function)
            plotting_utilities.plot_all_parameters_average(averaged_data, average_multiple_plots, activation_function)

    # Final values for each seed
    final_values_file = os.path.join(directory, f'{activation_function}_final_values_data.csv')

    if not os.path.exists(final_values_file):
        raw_data = pd.read_csv(raw_data_file)
        final_epoch_data = raw_data[raw_data['Seed & Epoch'].str.contains("Epoch 50000")]
        final_epoch_data.to_csv(final_values_file, index=False)


for activation_function in activation_functions:
    run_for_activation(activation_function)