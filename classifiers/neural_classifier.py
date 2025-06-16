"""
TDA Neural Network Classifier for Semantic Entailment Detection - CORRECTED VERSION

This module implements a neural network classifier that uses geometric features
combined with REAL TDA perturbation analysis to classify entailment relationships.

Architecture: 16 input features → 128 → 64 → 32 → 16 → 3 output classes (with softmax)
Features: Per-example geometric + REAL TDA perturbation changes + spatial context
"""

import logging

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from Cython.Shadow import returns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
import ripser
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TDAFeatureExtractor:
    """
        Extract features combining per-example geometric data with REAL TDA perturbation analysis.

        Features extracted (15 total):
        1-3: Per-example geometric (cone energy, order energy, hyperbolic distance)
        4-7: Spatial context (local density + distances to 3 class centroids)
        8-16: REAL TDA perturbation features (persistence/betti/significant changes × 3 classes)
    """

    def __init__(self, k_neighbors: int = 5):
        """
        Initialize the feature extractor.

        Args:
            k_neighbors: Number of neighbors for local density computation
        """
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Will be loaded from neural network data
        self.point_clouds = {}
        self.class_statistics = {}
        self.tda_params = {}
        self.class_centroids = {}
        self.training_geometric_features = None
        self.training_labels = None

        # Feature names for reference
        self.feature_names = [
            # Core geometric features (3)
            'cone_energy',
            'order_energy',
            'hyperbolic_distance',
            # Spatial features (4)
            'local_density',
            'dist_to_entailment_centroid',
            'dist_to_neutral_centroid',
            'dist_to_contradiction_centroid',
            # REAL TDA perturbation features (9)
            'persistence_change_entailment',
            'persistence_change_neutral',
            'persistence_change_contradiction',
            'betti_change_entailment',
            'betti_change_neutral',
            'betti_change_contradiction',
            'significant_change_entailment',
            'significant_change_neutral',
            'significant_change_contradiction'
        ]

    def fit(self, training_data: Dict):
        """
        Fit the feature extractor on neural network training data.

        Expected training_data structure (from updated tda_integration):
        {
            'cone_violations': tensor of shape [n_samples, 3] containing
                              [cone_energy, order_energy, hyperbolic_distance] per sample
            'labels': list of string labels ['entailment', 'neutral', 'contradiction']
            'point_clouds': dict with clean point clouds for each class
            'class_statistics': dict with per-example statistics for each class
            'tda_params': TDA parameters (maxdim, thresh, coeff)
        }
        """
        logger.info("Fitting TDA feature extractor with REAL perturbation analysis...")

        # Extract geometric features (cone_energy, order_energy, hyperbolic_distance)
        if isinstance(training_data['cone_violations'], torch.Tensor):
            self.training_geometric_features = training_data['cone_violations'].numpy()
        else:
            self.training_geometric_features = np.array(training_data['cone_violations'])

        self.training_labels = training_data['labels']

        # Load TDA data for perturbation analysis
        self.point_clouds = training_data['point_clouds']
        self.class_statistics = training_data['class_statistics']
        self.tda_params = training_data['tda_params']

        # Compute class centroids in geometric space
        self._compute_class_centroids()

        # Extract features for training data to fit scaler
        logger.info("Extracting features for training samples (this may take a while due to TDA computations)...")
        training_features = []

        for i in range(len(training_data['labels'])):
            if i % 100 == 0:
                logger.info(f"Processing training sample {i}/{len(training_data['labels'])}")

            sample_data = {
                'geometric_features': self.training_geometric_features[i]  # [cone, order, hyperbolic]
            }
            features = self._extract_single_sample_features(sample_data)
            training_features.append(features)

        training_features = np.array(training_features)

        # Fit scaler
        self.scaler.fit(training_features)
        self.is_fitted = True

        logger.info(f"Feature extractor fitted. Feature dimension: {training_features.shape[1]}")
        logger.info(f"Class centroids computed for: {list(self.class_centroids.keys())}")
        logger.info(f"Point clouds available for: {list(self.point_clouds.keys())}")

    def transform(self, sample_data: Dict) -> np.ndarray:
        """
        Transform a single sample into feature vector.

        Args:
            sample_data: Dictionary containing:
                - geometric_features: [cone_energy, order_energy, hyperbolic_distance]

        Returns:
            Normalized feature vector
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")

        features = self._extract_single_sample_features(sample_data)
        features = features.reshape(1, -1)
        normalized_features = self.scaler.transform(features)

        return normalized_features.flatten()

    def fit_transform(self, training_data: Dict) -> np.ndarray:
        """
        Fit extractor and transform training data.

        Args:
            training_data: Training data dictionary

        Returns:
            Normalized feature matrix [n_samples, n_features]
        """
        self.fit(training_data)

        # Transform all training samples
        logger.info("Transforming all training samples...")
        features_matrix = []
        for i in range(len(training_data['labels'])):
            if i % 100 == 0:
                logger.info(f"Transforming sample {i}/{len(training_data['labels'])}")

            sample_data = {
                'geometric_features': self.training_geometric_features[i]
            }
            features = self.transform(sample_data)
            features_matrix.append(features)

        return np.array(features_matrix)

    def _compute_class_centroids(self):
        """Compute centroid for each class in geometric feature space."""
        labels = self.training_labels

        for class_name in ['entailment', 'neutral', 'contradiction']:
            class_mask = np.array(labels) == class_name
            if np.any(class_mask):
                class_features = self.training_geometric_features[class_mask]
                centroid = np.mean(class_features, axis=0)
                self.class_centroids[class_name] = centroid
                logger.info(f"Computed centroid for {class_name}: {len(class_features)} samples")

    def _extract_single_sample_features(self, sample_data: Dict) -> np.ndarray:
        """Extract all features for a single sample."""
        features = []
        geometric_features = sample_data['geometric_features']

        # Features 1-3: Core geometric features
        features.extend(geometric_features.tolist() if hasattr(geometric_features, 'tolist')
                        else list(geometric_features))

        # Feature 4: Local density (distance-weighted k-NN)
        local_density = self._compute_local_density(geometric_features)
        features.append(local_density)

        # Features 5-7: Distances to class centroids
        for class_name in ['entailment', 'neutral', 'contradiction']:
            if class_name in self.class_centroids:
                distance = np.linalg.norm(geometric_features - self.class_centroids[class_name])
                features.append(distance)
            else:
                features.append(0.0)  # Fallback

        # Features 8-16: REAL TDA perturbation features
        tda_perturbation_features = self._compute_real_tda_perturbation_features(geometric_features)
        features.extend(tda_perturbation_features)

        return np.array(features)






    def _extract_single_sample_features(self, sample_data: Dict) -> np.ndarray:
        """Extract all features for a single sample."""
        features = []

        # Features 1-2: Core geometric features
        features.append(float(sample_data['cone_energy']))
        features.append(float(sample_data['order_energy']))

        # Feature 3: Local density (distance-weighted k-NN)
        local_density = self._compute_local_density(sample_data['cone_violations'])
        features.append(local_density)

        # Features 4-6: Distances to class centroids
        for class_name in ['entailment', 'neutral', 'contradiction']:
            if class_name in self.class_centroids:
                distance = np.linalg.norm(
                    sample_data['cone_violations'] - self.class_centroids[class_name]
                )
                features.append(distance)
            else:
                features.append(0.0)  # Fallback if centroid not available

        # Features 7-12: TDA fit scores
        tda_fit_scores = self._compute_tda_fit_scores(sample_data['cone_violations'])
        features.extend(tda_fit_scores)

        return np.array(features)

    def _compute_local_density(self, geometric_features: np.ndarray) -> float:
        """Compute distance-weighted k-NN density in geometric space."""
        if self.training_geometric_features is None:
            return 0.0

        # Find k nearest neighbors in geometric space
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        nbrs.fit(self.training_geometric_features)

        distances, indices = nbrs.kneighbors(geometric_features.reshape(1, -1))
        distances = distances.flatten()

        # Distance-weighted density (avoid division by zero)
        weights = 1.0 / (distances + 1e-8)
        density = np.sum(weights)

        return density


    def _compute_real_tda_perturbation_features(self, geometric_features: np.ndarray) -> List[float]:
        """
        Compute REAL TDA perturbation features using actual ripser computation.

        This is the core of our approach:
        1. Add test sample to each class point cloud
        2. Recompute TDA using ripser
        3. Measure actual changes in per-example statistics

        Returns 9 features: [persistence_change×3, betti_change×3, significant_change×3]
        """
        perturbation_features = []

        for class_name in ['entailment', 'neutral', 'contradiction']:
            if class_name not in self.point_clouds or class_name not in self.class_statistics:
                # Fallback if data not available
                perturbation_features.extend([0.0, 0.0, 0.0])  # persistence, betti, significant
                continue

            # Get original point cloud and statistics
            original_cloud = self.point_clouds[class_name]
            original_stats = self.class_statistics[class_name]

            try:
                # Add test sample to point cloud
                augmented_cloud = np.vstack([original_cloud, geometric_features.reshape(1, -1)])

                # Compute TDA for augmented cloud using ripser
                result = ripser.ripser(augmented_cloud, **self.tda_params)
                diagrams = result['dgms']

                # Extract features from augmented cloud (reuse extraction logic)
                augmented_features = self._extract_tda_features_from_diagrams(diagrams)

                # Compute new per-example statistics
                new_n_points = len(augmented_cloud)
                new_total_persistence_per_example = augmented_features['total_persistence'] / new_n_points
                new_betti_sum_per_example = augmented_features['betti_sum'] / new_n_points
                new_significant_features_per_example = augmented_features['n_significant_features'] / new_n_points

                # Compute changes (absolute differences)
                persistence_change = abs(
                    new_total_persistence_per_example - original_stats['total_persistence_per_example']
                )
                betti_change = abs(
                    new_betti_sum_per_example - original_stats['betti_sum_per_example']
                )
                significant_change = abs(
                    new_significant_features_per_example - original_stats['significant_features_per_example']
                )

                perturbation_features.extend([persistence_change, betti_change, significant_change])

            except Exception as e:
                logger.warning(f"TDA computation failed for {class_name}: {e}")
                # Use zero changes as fallback
                perturbation_features.extend([0.0, 0.0, 0.0])

        return perturbation_features

    def _extract_tda_features_from_diagrams(self, diagrams: List[np.ndarray]) -> Dict:
        """Extract TDA features from persistence diagrams (simplified version)."""
        total_persistence = 0.0
        betti_numbers = []
        all_lifespans = []

        for dim, diagram in enumerate(diagrams):
            betti_numbers.append(len(diagram))

            if len(diagram) > 0:
                # Extract finite features only
                finite_mask = np.isfinite(diagram[:, 1])
                finite_diagram = diagram[finite_mask]

                if len(finite_diagram) > 0:
                    lifespans = finite_diagram[:, 1] - finite_diagram[:, 0]
                    valid_lifespans = lifespans[lifespans > 0]

                    if len(valid_lifespans) > 0:
                        total_persistence += np.sum(valid_lifespans)
                        all_lifespans.extend(valid_lifespans)

        # Compute significant features
        if len(all_lifespans) > 0:
            mean_lifespan = np.mean(all_lifespans)
            n_significant_features = np.sum(np.array(all_lifespans) > mean_lifespan)
        else:
            n_significant_features = 0

        return {
            'total_persistence': total_persistence,
            'betti_sum': sum(betti_numbers),
            'n_significant_features': int(n_significant_features),
            'betti_numbers': betti_numbers
        }


class TDANeuralClassifier(nn.Module):
    """
    Neural network classifier using geometric + REAL TDA perturbation features.

    Architecture: 15 → 128 → 64 → 32 → 16 → 3 (raw logits for CrossEntropyLoss)
    """

    def __init__(self, input_dim: int = 15, dropout_rate: float = 0.3):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features (15 for full feature set)
            dropout_rate: Dropout probability for regularization
        """
        super(TDANeuralClassifier, self).__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # Network layers following architecture document
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Output layer - raw logits (no activation for CrossEntropyLoss)
        self.output = nn.Linear(16, 3)  # 3 classes: entailment, neutral, contradiction

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Raw logits [batch_size, 3] - use with CrossEntropyLoss
        """
        # Layer 1: 128 neurons
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        # Layer 2: 64 neurons
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        # Layer 3: 32 neurons
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout3(x)

        # Layer 4: 16 neurons
        x = self.layer4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.dropout4(x)

        # Output layer - raw logits (no activation)
        x = self.output(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities (applies softmax to logits)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probabilities = self.predict_proba(x)
        return torch.argmax(probabilities, dim=1)


def load_neural_network_data(data_path: str) -> Dict:
    """
    Load data from updated TDA integration for neural network training.

    Expected file: 'results/tda_integration/neural_network_data.pt'
    """
    logger.info(f"Loading neural network data from {data_path}")

    try:
        data = torch.load(data_path, map_location='cpu')

        # Validate required fields
        required_fields = ['cone_violations', 'labels', 'point_clouds', 'class_statistics', 'tda_params']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        logger.info(f"Successfully loaded neural network data")
        logger.info(f"Samples: {len(data['labels'])}")
        logger.info(f"Point clouds: {list(data['point_clouds'].keys())}")
        logger.info(f"Class statistics: {list(data['class_statistics'].keys())}")

        # Print point cloud sizes
        for class_name, cloud in data['point_clouds'].items():
            logger.info(f"  {class_name} point cloud: {cloud.shape}")

        return data

    except Exception as e:
        logger.error(f"Failed to load neural network data from {data_path}: {e}")
        raise




def create_classifier_from_neural_data(
        data_path: str = "results/tda_integration/neural_network_data.pt") -> Tuple[TDANeuralClassifier, TDAFeatureExtractor]:
    """
    Create and initialize classifier from neural network data.

    Args:
        data_path: Path to neural network data file
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (classifier, feature_extractor)
    """
    # Load data
    training_data = load_neural_network_data(data_path)

    # Initialize feature extractor and fit on training data
    logger.info("Initializing feature extractor with REAL TDA perturbation analysis...")
    feature_extractor = TDAFeatureExtractor()
    features_matrix = feature_extractor.fit_transform(training_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create classifier with softmax output
    input_dim = features_matrix.shape[1]
    classifier = TDANeuralClassifier(input_dim=input_dim)
    classifier = classifier.to(device)

    logger.info(f"Created classifier with {input_dim} input features on device {device}")
    logger.info("Network outputs raw logits for use with CrossEntropyLoss")

    return classifier, feature_extractor


def prepare_training_data(
        features_matrix: np.ndarray,
        labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training data for PyTorch training.

    Args:
        features_matrix: Normalized feature matrix from feature extractor
        labels: String labels

    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    # Convert features to tensor
    X = torch.FloatTensor(features_matrix)

    # Convert string labels to numeric
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    numeric_labels = [label_to_idx[label] for label in labels]
    y = torch.LongTensor(numeric_labels)

    logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: {torch.bincount(y)}")

    return X, y


if __name__ == "__main__":
    # Test the implementation
    logger.info("Testing TDA Neural Classifier with REAL perturbation analysis...")

    # Test with neural network data
    neural_data_path = "results/tda_integration/neural_network_data.pt"

    try:
        classifier, feature_extractor = create_classifier_from_neural_data(neural_data_path)
        logger.info("✓ Successfully created classifier and feature extractor")

        # Test forward pass with dummy data
        batch_size = 16  # Smaller batch for testing
        input_dim = classifier.input_dim
        dummy_input = torch.randn(batch_size, input_dim)

        # Test forward pass (raw logits)
        output = classifier(dummy_input)
        predictions = classifier.predict(dummy_input)

        logger.info(f"✓ Forward pass successful: {output.shape}")
        logger.info(f"✓ Output is raw logits (no activation): range [{output.min():.4f}, {output.max():.4f}]")
        logger.info(f"✓ Predictions shape: {predictions.shape}")

        # Test with CrossEntropyLoss (correct for raw logits)
        criterion = nn.CrossEntropyLoss()
        dummy_labels = torch.randint(0, 3, (batch_size,))
        loss = criterion(output, dummy_labels)

        logger.info(f"✓ CrossEntropyLoss test successful: {loss.item():.4f}")
        logger.info("✓ Network is ready for CrossEntropyLoss training!")

        logger.info("TDA Neural Classifier implementation test completed successfully!")

        # Test a single sample transformation
        if hasattr(feature_extractor, 'training_geometric_features'):
            test_sample = feature_extractor.training_geometric_features[0]
            test_sample_data = {'geometric_features': test_sample}

            logger.info("Testing single sample transformation with REAL TDA perturbation...")
            features = feature_extractor.transform(test_sample_data)
            logger.info(f"✓ Single sample features extracted: {len(features)} features")
            logger.info(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")

    except FileNotFoundError:
        logger.warning(f"Neural network data file not found at {neural_data_path}")









