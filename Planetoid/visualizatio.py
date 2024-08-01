import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    # Load beta values
    beta_values = []
    data_name = "Photo"
    with open(f'beta_values_{data_name.lower()}.txt', 'r') as f:
        for line in f:
            # epoch, layer, value = line.strip().split(' ')
            new_line = line.strip().split(' ')
            epoch, layer, value = [element for element in new_line if element]
            beta_values.append((int(epoch), int(layer), float(value)))

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
    plt.savefig(f'beta_values_{data_name.lower()}.png')

    # If you still want to display the plot, you can uncomment the following line
    # plt.show()

if __name__=="__main__":
    main()