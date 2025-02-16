import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from DataLoader import loadHAD

def extract_embeddings(model, dataloader):
    embeddings = []
    labels = []
    model.eval()
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            embeddings.append(outputs)
            labels.append(targets.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def extract_features(dataset):
    features = []
    labels = []
    print("Extracting features...")
    for i in range(len(dataset)):
        data, label = dataset[i]
        last_time_step_feature = data[-1].numpy()
        features.append(last_time_step_feature)
        labels.append(label.numpy())
    print(len(features))
    return np.array(features), np.array(labels)



if __name__ == "__main__":
    print("Loading model...")
    model = torch.load('save/model/HAD_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_features_folder = r'data/HAD/test/input'
    test_labels_folder = r'data/HAD/test/label'
    dataset = loadHAD(test_features_folder, test_labels_folder, window_size=128, step_size=1)
    batch_size = 128
    print("Loading data...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings, labels = extract_embeddings(model, dataloader)

    print("Training t-SNE for embeddings...")
    tsne = TSNE(n_components=2, random_state=31, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    cmap = plt.get_cmap('tab20')

    colors = cmap(np.arange(20))

    np.random.seed(42)  
    selected_colors_indices = np.random.choice(20, 12, replace=False)
    colors = colors[selected_colors_indices]
    
    for i, color in enumerate(colors):
        indices = labels == i
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color, s=5, label=f'Class {i}')

    plt.axis('off') 
    plt.savefig('save/HAD/TSNE_Embed.png')
    print("Embeddings visualization saved as 'TSNE_Embed.png'")

    print("Extracting input features for t-SNE...")
    features, labels = extract_features(dataset)

    print("Training t-SNE for input features...")
    tsne_input = TSNE(n_components=2, random_state=32, verbose=1)
    features_2d = tsne_input.fit_transform(features)

    plt.figure(figsize=(10, 8))

    for i, color in enumerate(colors):
        indices = labels == i
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], c=color, s=5, label=f'Class {i}')

    plt.axis('off')  
    plt.savefig('save/HAD/TSNE_Input_Features.png')
    print("Input features visualization saved as 'TSNE_Input_Features.png'")