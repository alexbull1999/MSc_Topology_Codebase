"""
======================================================================
Advanced Topological and Geometric Evaluation Tools
======================================================================
This module provides a set of advanced tools for evaluating the quality
of a learned latent space, focusing on methods that are aware of
complex, non-linear, and topological structures.

Includes:
- HDBSCANClustering: A powerful density-based clustering algorithm.
- ToMAToClustering: A clustering method explicitly based on 0D persistent homology.
- PersistenceImageClassifier: A sophisticated classifier that uses the local
  topology of a point's neighborhood as a feature.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from data_loader_global import GlobalDataLoader
from attention_autoencoder_model import AttentionAutoencoder

# Import specialized libraries
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage
    GIOTTO_TDA_AVAILABLE = True
except ImportError:
    GIOTTO_TDA_AVAILABLE = False


def _map_clusters_to_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Finds the optimal mapping between cluster labels and true class labels
    using the Hungarian algorithm to maximize accuracy.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        pred_labels (np.ndarray): The predicted cluster labels.

    Returns:
        np.ndarray: A new array of predicted labels, re-labeled to match the true labels.
    """
    # Create a confusion matrix (contingency matrix)
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Use the Hungarian algorithm to find the optimal assignment (permutation)
    # The algorithm finds the assignment that maximizes the sum of the diagonal,
    # which corresponds to maximizing the number of correctly classified samples.
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    
    # Create the mapping from old cluster label to new (true) label
    mapping = {original: target for original, target in zip(col_ind, row_ind)}
    
    # Create the new re-labeled array
    relabeled_preds = np.array([mapping.get(label, -1) for label in pred_labels]) # Use -1 for unmapped
    
    return relabeled_preds


# --- 1. HDBSCAN CLUSTERING ---
class HDBSCANClustering:
    """
    Evaluates clustering performance using HDBSCAN, a density-based algorithm
    that is excellent at finding clusters of arbitrary shapes.
    """
    def __init__(self, min_cluster_size=5, min_samples=None):
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN is not installed. Please run 'pip install hdbscan'.")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        print(f"HDBSCANClustering Initialized (min_cluster_size={min_cluster_size})")

    def evaluate(self, latent_features: np.ndarray, true_labels: np.ndarray):
        """
        Performs HDBSCAN clustering and evaluates the results.
        """
        print("\n--- Starting HDBSCAN Evaluation ---")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples,
                                    core_dist_n_jobs=-1)
        
        print("Fitting HDBSCAN clusterer...")
        cluster_labels = clusterer.fit_predict(latent_features)
        
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) < 2:
            print("Warning: HDBSCAN found no significant clusters.")
            return {"accuracy": 0, "adjusted_rand_score": 0, "silhouette_score": -1}

        filtered_preds = cluster_labels[non_noise_mask]
        filtered_true = true_labels[non_noise_mask]
        filtered_features = latent_features[non_noise_mask]
        
        # --- NEW: Calculate Clustering Accuracy ---
        relabeled_preds = _map_clusters_to_labels(filtered_true, filtered_preds)
        accuracy = accuracy_score(filtered_true, relabeled_preds)
        
        ari = adjusted_rand_score(filtered_true, filtered_preds)
        silhouette = silhouette_score(filtered_features, filtered_preds)
        
        print(f"HDBSCAN found {len(np.unique(filtered_preds))} clusters.")
        print(f"Clustering Accuracy (mapped): {accuracy:.4f}")
        print(f"Adjusted Rand Score (ARI): {ari:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        
        return {
            "accuracy": accuracy,
            "adjusted_rand_score": ari,
            "silhouette_score": silhouette,
            "num_clusters_found": len(np.unique(filtered_preds))
        }


# --- 3. PERSISTENCE IMAGE CLASSIFIER (Unchanged) ---
class PersistenceImageClassifier:
    """
    A complete pipeline for training and evaluating a simple CNN classifier
    on persistence images generated from the local neighborhoods of a latent space.
    """
    
    class SimpleCNN(nn.Module):
        """A simple CNN for classifying 2D persistence images."""
        def __init__(self, num_classes=3):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            if x.ndim == 5:
                # Squeeze the third dimension: [64, 1, 1, 28, 28] -> [64, 1, 28, 28]
                x = x.squeeze(2)

            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x

    def __init__(self, k_neighbors=200, image_size=(28, 28), num_classes=3, device='cuda'):
        if not GIOTTO_TDA_AVAILABLE:
            raise ImportError("giotto-tda is not installed. Please run 'pip install giotto-tda'.")
        self.k = k_neighbors
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.SimpleCNN(num_classes).to(self.device)
        self.persistence_computer = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=-1)
        self.image_transformer = PersistenceImage(n_bins=self.image_size[0])
        
        print("PersistenceImageClassifier Initialized.")
        print(f"  Local neighborhood size (k): {self.k}")
        print(f"  Persistence image resolution: {self.image_size[0]}x{self.image_size[1]}")

    def create_persistence_images(self, reference_cloud: np.ndarray, target_points: np.ndarray):
        """
        Creates persistence images for each point in `target_points` by finding
        its k-nearest neighbors in the `reference_cloud`.
        """
        print(f"Creating persistence images for {len(target_points)} points...")
        start_time = time.time()
        
        # 1. Find the k-nearest neighbors for each target point in the reference cloud
        print(f"  > Finding {self.k}-nearest neighbors in reference cloud of shape {reference_cloud.shape}...")
        nn_finder = NearestNeighbors(n_neighbors=self.k, algorithm='auto', n_jobs=-1)
        nn_finder.fit(reference_cloud)
        _, neighbor_indices = nn_finder.kneighbors(target_points)
        print("  > Neighbor search complete.")

        # 2. Build the local neighborhood point clouds
        local_neighborhoods = []
        for i, point in enumerate(target_points):
            # The neighborhood includes the point itself + its k neighbors
            neighborhood_points = np.vstack([point, reference_cloud[neighbor_indices[i]]])
            local_neighborhoods.append(neighborhood_points)
        
        # 3. Compute persistence diagrams for all neighborhoods at once
        # The input shape needs to be (n_samples, n_points, n_features)
        print("  > Computing persistence diagrams for all neighborhoods...")
        diagrams = self.persistence_computer.fit_transform(np.array(local_neighborhoods))
        
        # 4. Convert diagrams to persistence images
        print("  > Converting diagrams to images...")
        images = self.image_transformer.fit_transform(diagrams)
        images_reshaped = images[:, np.newaxis, :, :]
        
        print(f"  > Created {len(images_reshaped)} images in {time.time() - start_time:.2f} seconds.")
        return torch.from_numpy(images_reshaped).float()

    def train(self, images: torch.Tensor, labels: torch.Tensor, epochs=50, batch_size=64):
        print("\n--- Training Persistence Image CNN ---")
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_images, batch_labels in loader:
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")
        print("CNN training complete.")

    def evaluate(self, images: torch.Tensor, labels: torch.Tensor, batch_size=64):
        print("\n--- Evaluating Persistence Image CNN ---")
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                outputs = self.model(batch_images.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch_labels.numpy())
        
        accuracy = accuracy_score(all_true, all_preds)
        print(f"CNN Classification Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}


# --- MAIN EXECUTION SCRIPT ---
def main():
    """
    Main function to run the full topological evaluation pipeline.
    """
    CONFIG = {
        'model_path': "entailment_surfaces/supervised_contrastive_autoencoder/experiments/topological_autoencoder_torchph_phase1_20250722_141618/checkpoints/best_model.pt",
        'train_data_path': "data/processed/snli_full_standard_SBERT.pt",
        'val_data_path': "data/processed/snli_full_standard_SBERT_validation.pt",
        
        'model_params': {
            'input_dim': 768,  # Adjust if different
            'latent_dim': 75,  # Adjust to match your trained model
            'hidden_dims': [1024, 768, 512, 256, 128],  # Adjust to match your model
            'dropout_rate': 0.2
        },
        
        'eval_params': {
            'num_samples_for_clustering': 10000,
            'num_samples_for_pi_cnn_train': 50000,
            'num_samples_for_pi_cnn_test': 3000
        }
    }

    print("="*60)
    print("STARTING ADVANCED TOPOLOGICAL EVALUATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the trained Autoencoder model
    print(f"Loading model from: {CONFIG['model_path']}")
        
    model = AttentionAutoencoder(**CONFIG['model_params']).to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device)['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 2. Load data using your GlobalDataLoader
    data_loader = GlobalDataLoader(
        train_path=CONFIG['train_data_path'],
        val_path=CONFIG['val_data_path'],
        test_path=CONFIG['val_data_path'], # Using validation as test set for this script
        embedding_type='lattice'
    )
    train_dataset, val_dataset, _ = data_loader.load_data()
    
    def get_all_data(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        batch = next(iter(loader))
        return batch['embeddings'], batch['labels']

    train_embeddings, train_labels = get_all_data(train_dataset)
    val_embeddings, val_labels = get_all_data(val_dataset)
    
    # 3. Generate Latent Representations
    print("\nGenerating latent representations...")
    with torch.no_grad():
        train_latent_torch = model.get_latent_representations(train_embeddings.to(device))
        val_latent_torch = model.get_latent_representations(val_embeddings.to(device))
    
    train_latent_np = train_latent_torch.cpu().numpy()
    val_latent_np = val_latent_torch.cpu().numpy()
    train_labels_np = train_labels.numpy()
    val_labels_np = val_labels.numpy()
    print(f"  > Train latent shape: {train_latent_np.shape}")
    print(f"  > Val latent shape:   {val_latent_np.shape}")

    # Subsample for clustering evaluations
    cluster_indices = np.random.choice(len(val_latent_np), min(len(val_latent_np), CONFIG['eval_params']['num_samples_for_clustering']), replace=False)
    
    # 4. Run Clustering Evaluations
    hdbscan_eval = HDBSCANClustering()
    hdbscan_results = hdbscan_eval.evaluate(val_latent_np[cluster_indices], val_labels_np[cluster_indices])
    
    # 5. Run Persistence Image Classification
    pi_classifier = PersistenceImageClassifier(k_neighbors=200, device=device)
    
    # Create images for our validation set by finding their neighbors in the full training set
    num_pi_train = min(CONFIG['eval_params']['num_samples_for_pi_cnn_train'], len(train_latent_np))
    pi_train_indices = np.random.choice(len(train_latent_np), num_pi_train, replace=False)
    pi_train_subset = train_latent_np[pi_train_indices]
    pi_train_labels_subset = train_labels_np[pi_train_indices]

    # The reference cloud for finding neighbors is the full training set
    pi_train_images = pi_classifier.create_persistence_images(
        reference_cloud=train_latent_np,
        target_points=pi_train_subset
    )

    num_pi_test = min(CONFIG['eval_params']['num_samples_for_pi_cnn_test'], len(val_latent_np))
    pi_test_indices = np.random.choice(len(val_latent_np), num_pi_test, replace=False)
    pi_test_subset = val_latent_np[pi_test_indices]
    pi_test_labels_subset = val_labels_np[pi_test_indices]
    
    # The reference cloud is STILL the full training set
    pi_test_images = pi_classifier.create_persistence_images(
        reference_cloud=train_latent_np,
        target_points=pi_test_subset
    )

    pi_classifier.train(pi_train_images, torch.from_numpy(pi_train_labels_subset).long(), epochs=50)
    cnn_results = pi_classifier.evaluate(pi_test_images, torch.from_numpy(pi_test_labels_subset).long())

    # 6. Print Final Summary
    print("\n" + "="*60)
    print("FINAL TOPOLOGICAL EVALUATION SUMMARY")
    print("="*60)
    print(f"\nHDBSCAN Results (on {CONFIG['eval_params']['num_samples_for_clustering']} samples):")
    print(json.dumps(hdbscan_results, indent=2))
    print(f"\nPersistence Image CNN Results (trained on {len(pi_train_images)}, tested on {len(pi_test_images)}):")
    print(json.dumps(cnn_results, indent=2))
    print("="*60)

if __name__ == "__main__":
    main()