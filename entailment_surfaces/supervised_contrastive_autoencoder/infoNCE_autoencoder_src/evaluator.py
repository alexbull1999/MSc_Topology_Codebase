"""
Evaluator for InfoNCE + Order Embeddings autoencoder
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans


class InfoNCEOrderEvaluator:
    """
    Evaluator for InfoNCE + Order Embeddings approach
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def extract_features(self, data_loader):
        """Extract latent features and labels"""
        self.model.eval()
        
        all_latent_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                premise_embeddings = batch['premise_embedding'].to(self.device)
                hypothesis_embeddings = batch['hypothesis_embedding'].to(self.device)
                labels = batch['label']
                
                # Concatenate premise and hypothesis embeddings for the model
                premise_hyp_concat = torch.cat([premise_embeddings, hypothesis_embeddings], dim=1)
                
                # Get latent features
                latent_features, _ = self.model(premise_hyp_concat)
                
                all_latent_features.append(latent_features.cpu())
                all_labels.append(labels)
        
        features = torch.cat(all_latent_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        return features, labels
    
    def evaluate(self, data_loader):
        """Full evaluation"""
        features, labels = self.extract_features(data_loader)
        
        # Clustering evaluation
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_predictions = kmeans.fit_predict(features)
        
        # Map clusters to true labels (best assignment)
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(labels, cluster_predictions)
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        # Create mapping
        cluster_to_label = {}
        for i, j in zip(row_ind, col_ind):
            cluster_to_label[j] = i
        
        # Apply mapping
        mapped_predictions = np.array([cluster_to_label[pred] for pred in cluster_predictions])
        clustering_accuracy = accuracy_score(labels, mapped_predictions)
        
        # Silhouette score
        silhouette = silhouette_score(features, labels)
        
        # Distance-based metrics
        pos_distances = []
        neg_distances = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                if labels[i] == labels[j]:
                    pos_distances.append(dist)
                else:
                    neg_distances.append(dist)
        
        separation_ratio = np.mean(neg_distances) / np.mean(pos_distances) if pos_distances else 0
        
        results = {
            'clustering_accuracy': clustering_accuracy,
            'silhouette_score': silhouette,
            'separation_ratio': separation_ratio,
            'pos_distance_mean': np.mean(pos_distances) if pos_distances else 0,
            'neg_distance_mean': np.mean(neg_distances) if neg_distances else 0,
            'num_samples': len(features)
        }
        
        return results