import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def main():
    # Load beta values
    data_name = "Pubmed"
    beta_values = []
    with open('beta_values_pubmed.txt', 'r') as f:
        for line in f:
            new_line = line.strip().split(' ')
            # print([element for element in new_line if element])
            epoch, layer, value = [element for element in new_line if element]
            beta_values.append((int(epoch), int(layer), float(value)))

    # Convert to numpy array for easier manipulation
    beta_values = np.array(beta_values)

    # Normalize layer indices for the gradient
    normalized_layers = beta_values[:, 1] / max(beta_values[:, 1])
    colors = cm.viridis(normalized_layers)  # Change colormap if desired

    plt.figure(figsize=(12, 6))
    plt.bar(beta_values[:, 1], beta_values[:, 2], color=colors)
    plt.xlabel('/th-layer')
    plt.ylabel(r'$\beta^{(l)}$')
    plt.title(f'Beta values at layer for {data_name} Dataset')

    # Save the figure as a PNG file
    plt.savefig(f'beta_values_layer_{data_name.lower()}.png')
    # plt.show()

if __name__ == "__main__":
    main()
