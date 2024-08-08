import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.colors import PowerNorm

def visualization(beta_values, data_name):
    # # Load beta values
    # beta_values = []
    # data_name = "Computers"
    # with open(f'beta_values_{data_name.lower()}.txt', 'r') as f:
    #     for line in f:
    #         # epoch, layer, value = line.strip().split(' ')
    #         new_line = line.strip().split(' ')
    #         epoch, layer, value = [element for element in new_line if element]
    #         beta_values.append((int(epoch), int(layer), float(value)))

    # Convert to numpy array for easier manipulation
    beta_values = np.array(beta_values)

    # Plot histograms for the final epoch
    final_epoch = max(beta_values[:, 0])
    final_beta_values = beta_values[beta_values[:, 0] == final_epoch]

    plt.figure(figsize=(12, 6))

    # Normalize layer indices for the gradient
    normalized_layers = final_beta_values[:, 1] / max(final_beta_values[:, 1])
    colors = cm.viridis(normalized_layers)  # Change colormap if desired

    plt.bar(final_beta_values[:, 1], final_beta_values[:, 2], color=colors)
    plt.xlabel('/th-layer')
    plt.ylabel(r'$\beta^{(l)}$')
    plt.title(f'Beta values at layer for {data_name}')

    # Save the figure as a PNG file
    plt.savefig(f'graph_results/beta_values_{data_name.lower()}.png')

    # If you still want to display the plot, you can uncomment the following line
    # plt.show()

def visualization_geom_fig(beta_values, graph_type, m, n, homo, hetero):
    # Convert to numpy array for easier manipulation
    beta_values = np.array(beta_values)

    # Plot histograms for the final epoch
    print(beta_values)
    final_epoch = max(beta_values[:, 0])
    final_beta_values = beta_values[beta_values[:, 0] == final_epoch]

    plt.figure(figsize=(20, 6))

    # Normalize the beta values for color mapping
    norm = PowerNorm(gamma=0.2, vmin=final_beta_values[:, 2].min(), vmax=final_beta_values[:, 2].max())
    colors = cm.Blues(norm(final_beta_values[:, 2]))

    # Create the bar plot
    plt.bar(final_beta_values[:, 1], final_beta_values[:, 2], color=colors)

    plt.xlabel('/th-layer')
    plt.ylabel(r'$\beta^{(l)}$')
    plt.title(f'Beta values at layer for {graph_type.upper()} with m={m} and n={n}')

    # Save the figure as a PNG file
    if homo == True:
        plt.savefig(f'graph_results/homo_beta_values_{graph_type.lower()}_{m}_{n}.png')
    elif hetero == True:
        plt.savefig(f'graph_results/hetero_beta_values_{graph_type.lower()}_{m}_{n}.png')

def matrix_visualization():
    number = 50
    first_part = [[f'beta_values_grid_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_hexagonal_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_kagome_{number}_{i}' for i in range(10, 35, 5)],
                 [f'beta_values_triangle_{number}_{i}' for i in range(10, 35, 5)],
                ]
    
    second_part = [[f'beta_values_grid_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_hexagonal_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_kagome_{number}_{i}' for i in range(35, 55, 5)],
                 [f'beta_values_triangle_{number}_{i}' for i in range(35, 55, 5)],
                ]

    for lst_names in [first_part, second_part]:
        if lst_names == first_part:
            num_cols = 5
            name = 'first_part'
        else:
            num_cols = 4
            name = 'second_part'
        # Flatten the list of lists
        flattened_lst_names = [item for sublist in lst_names for item in sublist]
        
        # Define the path to the directory containing the images
        image_dir = 'graph_results/'

        # Number of rows and columns for the plot
        num_rows = 4

        # Create a figure with the specified number of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop through the sorted graph names and plot each image
        for i, graph_name in enumerate(flattened_lst_names):
            # Construct the full path to the image
            img_path = f'{image_dir}/{graph_name}.png'
            
            # Load the image
            img = mpimg.imread(img_path)
            gr_name = graph_name.split('_')
            # Plot the image in the corresponding subplot
            axes[i].imshow(img)
            axes[i].set_title(f'{gr_name[2][0].upper() + gr_name[2][1:]} with m={gr_name[3]} n={gr_name[4]}' )
            axes[i].axis('off')  # Turn off axis

        # Adjust layout
        plt.tight_layout()
        # Save the figure as a PNG file
        plt.savefig(f'graph_results/matrix_graph_{number}_{name}.png')

# plt.show()
if __name__=="__main__":
    matrix_visualization()