import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import warnings

class PersistencePrototypeCreator:
    """
    Creates prototype persistence diagrams for each class based on the similarity analysis.
    
    Given the excellent stability results (CV < 0.05), we can confidently create
    representative prototypes for regularization.
    """
    
    def __init__(self, diagrams_data: Dict):
        self.diagrams_data = diagrams_data
        self.prototypes = {}
        print("Initialized prototype creator")
        print("Based on analysis results, all classes show excellent stability for averaging")
    
    def _clean_diagram(self, diagram: np.ndarray) -> np.ndarray:
        """Remove infinite and invalid points from persistence diagram"""
        if diagram.size == 0:
            return np.array([]).reshape(0, 2)
        
        # Remove infinite points
        finite_mask = np.isfinite(diagram).all(axis=1)
        clean_diagram = diagram[finite_mask]
        
        # Remove points where death <= birth (shouldn't happen but just in case)
        if clean_diagram.size > 0:
            valid_mask = clean_diagram[:, 1] > clean_diagram[:, 0]
            clean_diagram = clean_diagram[valid_mask]
        
        return clean_diagram
    
    def _compute_centroid_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 1: Simple centroid-based averaging
        Works well when diagrams have similar structure (which your analysis confirms)
        """
        print("    Using centroid-based averaging...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        
        # Remove empty diagrams
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if not non_empty_diagrams:
            return np.array([]).reshape(0, 2)
        
        # Find common number of features (use median)
        feature_counts = [len(d) for d in non_empty_diagrams]
        target_features = int(np.median(feature_counts))
        
        print(f"      Target features: {target_features} (median of {np.min(feature_counts)}-{np.max(feature_counts)})")
        
        # Pad or truncate diagrams to common size
        normalized_diagrams = []
        for diagram in non_empty_diagrams:
            if len(diagram) >= target_features:
                # Sort by persistence and take top features
                persistences = diagram[:, 1] - diagram[:, 0]
                top_indices = np.argsort(persistences)[-target_features:]
                normalized_diagrams.append(diagram[top_indices])
            else:
                # Pad with zeros (will be filtered out later)
                padding = np.zeros((target_features - len(diagram), 2))
                normalized_diagrams.append(np.vstack([diagram, padding]))
        
        # Compute centroid
        if normalized_diagrams:
            centroid = np.mean(normalized_diagrams, axis=0)
            # Remove zero-persistence points
            valid_mask = centroid[:, 1] > centroid[:, 0]
            centroid = centroid[valid_mask]
            return centroid
        else:
            return np.array([]).reshape(0, 2)
    
    def _compute_medoid_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 2: Medoid-based prototype (most representative diagram)
        More robust to outliers
        """
        print("    Using medoid-based selection...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if len(non_empty_diagrams) <= 1:
            return non_empty_diagrams[0] if non_empty_diagrams else np.array([]).reshape(0, 2)
        
        # Compute simple signature distances (fast approximation)
        signatures = []
        for diagram in non_empty_diagrams:
            if diagram.size > 0:
                persistences = diagram[:, 1] - diagram[:, 0]
                sig = np.array([
                    np.sum(persistences),
                    np.mean(persistences),
                    np.std(persistences),
                    len(persistences)
                ])
                signatures.append(sig)
            else:
                signatures.append(np.zeros(4))
        
        # Find medoid (diagram with minimum average distance to all others)
        signatures = np.array(signatures)
        distances = pdist(signatures, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Find index of medoid
        avg_distances = np.mean(distance_matrix, axis=1)
        medoid_idx = np.argmin(avg_distances)
        
        print(f"      Selected medoid: diagram {medoid_idx} with avg distance {avg_distances[medoid_idx]:.3f}")
        
        return non_empty_diagrams[medoid_idx]
    
    def _compute_robust_average_prototype(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Method 3: Robust averaging with outlier removal
        Best approach given your high stability
        """
        print("    Using robust averaging with outlier removal...")
        
        # Clean all diagrams
        cleaned_diagrams = [self._clean_diagram(d) for d in diagrams]
        non_empty_diagrams = [d for d in cleaned_diagrams if d.size > 0]
        
        if not non_empty_diagrams:
            return np.array([]).reshape(0, 2)
        
        # Compute signatures for outlier detection
        signatures = []
        for diagram in non_empty_diagrams:
            persistences = diagram[:, 1] - diagram[:, 0]
            sig = np.array([
                np.sum(persistences),
                np.mean(persistences),
                np.std(persistences),
                len(persistences),
                np.max(persistences)
            ])
            signatures.append(sig)
        
        signatures = np.array(signatures)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(signatures, 25, axis=0)
        Q3 = np.percentile(signatures, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find inliers
        inlier_mask = np.all((signatures >= lower_bound) & (signatures <= upper_bound), axis=1)
        inlier_diagrams = [non_empty_diagrams[i] for i in range(len(non_empty_diagrams)) if inlier_mask[i]]
        
        print(f"      Kept {len(inlier_diagrams)}/{len(non_empty_diagrams)} diagrams after outlier removal")
        
        if not inlier_diagrams:
            inlier_diagrams = non_empty_diagrams  # Fallback if all removed
        
        # Now compute centroid on inliers
        feature_counts = [len(d) for d in inlier_diagrams]
        target_features = int(np.median(feature_counts))
        
        # Normalize to common size
        normalized_diagrams = []
        for diagram in inlier_diagrams:
            if len(diagram) >= target_features:
                persistences = diagram[:, 1] - diagram[:, 0]
                top_indices = np.argsort(persistences)[-target_features:]
                normalized_diagrams.append(diagram[top_indices])
            else:
                # Pad with the most persistent feature repeated
                if len(diagram) > 0:
                    most_persistent = diagram[np.argmax(diagram[:, 1] - diagram[:, 0])]
                    padding = np.tile(most_persistent, (target_features - len(diagram), 1))
                    normalized_diagrams.append(np.vstack([diagram, padding]))
        
        # Compute robust centroid
        if normalized_diagrams:
            centroid = np.mean(normalized_diagrams, axis=0)
            valid_mask = centroid[:, 1] > centroid[:, 0]
            centroid = centroid[valid_mask]
            return centroid
        else:
            return np.array([]).reshape(0, 2)
    
    def create_prototypes(self, method: str = 'robust') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create prototype persistence diagrams for each class
        
        Args:
            method: 'centroid', 'medoid', or 'robust' (recommended)
        
        Returns:
            Dictionary with prototypes for each class and dimension
        """
        print(f"\nCreating prototypes using {method} method...")
        print("="*50)
        
        prototypes = {}
        
        for class_name, data in self.diagrams_data.items():
            print(f"\n--- Creating prototypes for {class_name.upper()} ---")
            
            class_prototypes = {}
            
            # Process H0 and H1 separately
            for dim_name in ['H0', 'H1']:
                print(f"  Processing {dim_name}...")
                diagrams = data[dim_name]
                
                if method == 'centroid':
                    prototype = self._compute_centroid_prototype(diagrams)
                elif method == 'medoid':
                    prototype = self._compute_medoid_prototype(diagrams)
                elif method == 'robust':
                    prototype = self._compute_robust_average_prototype(diagrams)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                class_prototypes[dim_name] = prototype
                print(f"    {dim_name} prototype: {len(prototype)} features")
                
                if len(prototype) > 0:
                    persistences = prototype[:, 1] - prototype[:, 0]
                    print(f"    Total persistence: {np.sum(persistences):.4f}")
                    print(f"    Max persistence: {np.max(persistences):.4f}")
            
            prototypes[class_name] = class_prototypes
        
        self.prototypes = prototypes
        return prototypes
    
    def save_prototypes(self, save_path: str):
        """Save prototypes to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.prototypes, f)
        print(f"\nPrototypes saved to {save_path}")
    
    def load_prototypes(self, load_path: str):
        """Load prototypes from file"""
        with open(load_path, 'rb') as f:
            self.prototypes = pickle.load(f)
        print(f"Prototypes loaded from {load_path}")
    
    def visualize_prototypes(self, save_path: str = None):
        """Create visualizations of the prototypes"""
        if not self.prototypes:
            print("No prototypes to visualize. Run create_prototypes() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Persistence Diagram Prototypes', fontsize=16)
        
        class_names = list(self.prototypes.keys())
        
        for dim_idx, dim_name in enumerate(['H0', 'H1']):
            for class_idx, class_name in enumerate(class_names):
                ax = axes[dim_idx, class_idx]
                
                prototype = self.prototypes[class_name][dim_name]
                
                if len(prototype) > 0:
                    # Plot persistence diagram
                    ax.scatter(prototype[:, 0], prototype[:, 1], alpha=0.7, s=50)
                    
                    # Plot diagonal line
                    max_val = np.max(prototype)
                    min_val = np.min(prototype)
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                    
                    ax.set_xlabel('Birth')
                    ax.set_ylabel('Death')
                    ax.set_title(f'{class_name.title()} {dim_name}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Empty\nDiagram', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{class_name.title()} {dim_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prototype visualization saved to {save_path}")
        
        plt.show()
    
    def get_prototype_summary(self) -> str:
        """Generate a summary report of the prototypes"""
        if not self.prototypes:
            return "No prototypes created yet."
        
        report = []
        report.append("="*60)
        report.append("PERSISTENCE DIAGRAM PROTOTYPES SUMMARY")
        report.append("="*60)
        
        for class_name, class_prototypes in self.prototypes.items():
            report.append(f"\n{class_name.upper()} CLASS PROTOTYPES:")
            
            for dim_name, prototype in class_prototypes.items():
                report.append(f"  {dim_name}:")
                
                if len(prototype) > 0:
                    persistences = prototype[:, 1] - prototype[:, 0]
                    report.append(f"    Features: {len(prototype)}")
                    report.append(f"    Total persistence: {np.sum(persistences):.4f}")
                    report.append(f"    Max persistence: {np.max(persistences):.4f}")
                    report.append(f"    Mean persistence: {np.mean(persistences):.4f}")
                else:
                    report.append(f"    Empty diagram")
        
        return "\n".join(report)


def main():
    """Main function to create prototypes"""
    print("Creating persistence diagram prototypes...")
    
    # Load the collected diagrams
    DIAGRAMS_PATH = 'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/collected_diagrams.pkl'
    
    if not Path(DIAGRAMS_PATH).exists():
        print(f"Error: {DIAGRAMS_PATH} not found!")
        return
    
    with open(DIAGRAMS_PATH, 'rb') as f:
        all_diagrams = pickle.load(f)
    
    # Create prototypes
    creator = PersistencePrototypeCreator(all_diagrams)
    
    # Use robust method (recommended given your stability results)
    methods = ['medoid', 'robust', 'centroid']
    for method in methods:
        prototypes = creator.create_prototypes(method=method)
    
        # Save prototypes
        PROTOTYPES_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototypes_{method}_vCosine.pkl'
        creator.save_prototypes(PROTOTYPES_PATH)
    
        # Generate summary
        summary = creator.get_prototype_summary()
        print("\n" + summary)
    
        # Save summary
        SUMMARY_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototype__{method}_summary_vCosine.txt'
        with open(SUMMARY_PATH, 'w') as f:
            f.write(summary)
    
        # Create visualizations
        VIZ_PATH = f'entailment_surfaces/supervised_contrastive_autoencoder/src/persistence_diagrams/prototype__{method}_visualizations_vCosine.png'
        creator.visualize_prototypes(VIZ_PATH)
    
        print(f"\nPrototype creation complete!")
        print(f"Files created:")
        print(f"  - Prototypes: {PROTOTYPES_PATH}")
        print(f"  - Summary: {SUMMARY_PATH}")
        print(f"  - Visualizations: {VIZ_PATH}")


if __name__ == '__main__':
    main()