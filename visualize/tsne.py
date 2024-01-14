import pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def main(feature_path):
    with open(feature_path, 'rb') as f:
        feature_banks = pickle.load(f)

    print(feature_banks.keys())
    args=feature_banks['args']
    train=feature_banks['training']
    test=feature_banks['testing']
    backdoor=feature_banks['backdoor']
    target=feature_banks['target']

    print('args',feature_banks['args'])
    print('train',feature_banks['training'].shape)
    print('test',feature_banks['testing'].shape)
    print('backdoor',feature_banks['backdoor'].shape)
    print('target',feature_banks['target'].shape)

    # Assuming test, backdoor, and target are numpy arrays of the shapes mentioned
    # For demonstration, I will create dummy data with the same shapes
    np.random.seed(0)  # For reproducibility of random data
    # Concatenating all arrays for TSNE
    num=500
    data = np.concatenate((test[:num], backdoor[:num], target))

    # Applying TSNE
    tsne = TSNE(n_components=2, random_state=0)
    transformed_data = tsne.fit_transform(data)

    # Splitting the transformed data
    transformed_test = transformed_data[:num]
    transformed_backdoor = transformed_data[num:num*2]
    transformed_target = transformed_data[-1]

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed_test[:, 0], transformed_test[:, 1], c='blue', label='Test-clean')
    plt.scatter(transformed_backdoor[:, 0], transformed_backdoor[:, 1], c='green', label='Backdoor')
    plt.scatter(transformed_target[0], transformed_target[1], c='red', label='Target', marker='x')
    plt.title("TSNE Visualization of Test, Backdoor, and Target Data")
    plt.legend()
    # plt.show()
    path = feature_path.split('/')[-2]
    plt.savefig(f'TSNE/{path}.png')
    plt.close()

if __name__ == '__main__':
    main('../output/stl10/cifar10_backdoored_encoder/2023-12-16-17:34:27bpp/feature_banks.pkl')