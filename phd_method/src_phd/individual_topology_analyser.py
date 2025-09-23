#!/usr/bin/env python3
"""
Individual Sample Topological Analysis

This script analyzes the persistence diagrams of successful entailment, neutral, and 
contradiction samples from SNLI and MNLI to understand the topological characteristics
that make them classifiable.

Creates persistence diagrams and analyzes topological features for qualitative examples.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from gph.python import ripser_parallel
from sklearn.metrics.pairwise import pairwise_distances

# Add paths for existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your existing point cloud generator
from point_cloud_clustering_test import SeparateModelPointCloudGenerator


@dataclass
class TopologicalAnalysis:
    """Results of topological analysis for one sample"""
    sample_id: str
    dataset: str
    class_name: str
    premise_text: str
    hypothesis_text: str
    
    # Point cloud statistics
    total_points: int
    premise_tokens: int
    hypothesis_tokens: int
    
    # Energy measurements
    forward_energy: float
    backward_energy: float
    asymmetric_energy: float
    
    # Topological features
    h0_features: int
    h1_features: int
    h0_max_persistence: float
    h1_max_persistence: float
    h0_total_persistence: float
    h1_total_persistence: float
    
    # Persistence diagrams
    persistence_diagrams: List[np.ndarray]


class IndividualTopologyAnalyzer:
    """
    Analyze the topology of individual premise-hypothesis pairs
    """
    
    def __init__(self, 
                 order_model_path: str,
                 asymmetry_model_path: str,
                 device: str = 'cuda'):
        
        self.device = device
        
        # Initialize the point cloud generator using your existing approach
        self.point_cloud_generator = SeparateModelPointCloudGenerator(
            order_model_path=order_model_path,
            asymmetry_model_path=asymmetry_model_path,
            hyperbolic_model_path=None,  # Optional
            device=device
        )
        
        print("Individual topology analyzer initialized")
    
    def load_dataset_samples(self, dataset_name: str, data_path: str, max_samples_per_class: int = 50) -> Dict:
        """
        Load samples from SNLI or MNLI datasets
        """
        print(f"Loading {dataset_name} samples from {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Organize by class
        class_samples = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for i, label in enumerate(data['labels']):
            if len(class_samples[label]) < max_samples_per_class:
                sample = {
                    'index': i,
                    'premise_tokens': data['premise_tokens'][i],
                    'hypothesis_tokens': data['hypothesis_tokens'][i],
                    'label': label
                }
                
                # Add text if available
                if 'premise_texts' in data:
                    sample['premise_text'] = data['premise_texts'][i]
                if 'hypothesis_texts' in data:
                    sample['hypothesis_text'] = data['hypothesis_texts'][i]
                
                class_samples[label].append(sample)
        
        print(f"Loaded samples per class:")
        for class_name, samples in class_samples.items():
            print(f"  {class_name}: {len(samples)}")
        
        return class_samples
    
    def analyze_sample_topology(self, sample: Dict, dataset_name: str, sample_id: str) -> TopologicalAnalysis:
        """
        Analyze the topological characteristics of one sample
        """
        premise_tokens = sample['premise_tokens']
        hypothesis_tokens = sample['hypothesis_tokens']
        
        print(f"Analyzing {sample_id} ({sample['label']})")
        print(f"  Tokens: P={premise_tokens.shape[0]}, H={hypothesis_tokens.shape[0]}")
        
        # Generate point cloud using your existing method
        point_cloud, stats = self.point_cloud_generator.generate_premise_hypothesis_point_cloud(
            premise_tokens, hypothesis_tokens
        )
        
        # Get model analysis (energies)
        model_analysis = self.point_cloud_generator.analyze_model_outputs(premise_tokens, hypothesis_tokens)
        
        print(f"  Point cloud: {stats['combined_total_points']} points")
        print(f"  Order energy: {model_analysis['order_model']['order_violation_energy']:.4f}")
        print(f"  Asymmetric energy: {model_analysis['asymmetry_model']['asymmetric_energy']:.4f}")
        
        # Compute distance matrix and persistence diagrams
        point_cloud_np = point_cloud.numpy()
        distance_matrix = pairwise_distances(point_cloud_np, metric='braycurtis')
        
        # Compute persistence diagrams
        diagrams = ripser_parallel(distance_matrix, maxdim=1, n_threads=-1, metric="precomputed")['dgms']
        
        # Analyze H0 and H1 features
        h0_diagram = diagrams[0]
        h1_diagram = diagrams[1]
        
        # Filter infinite persistence
        h0_finite = h0_diagram[np.isfinite(h0_diagram).all(axis=1)]
        h1_finite = h1_diagram[np.isfinite(h1_diagram).all(axis=1)]
        
        # Calculate statistics
        h0_lifespans = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([])
        h1_lifespans = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([])
        
        # Create analysis object
        analysis = TopologicalAnalysis(
            sample_id=sample_id,
            dataset=dataset_name,
            class_name=sample['label'],
            premise_text=sample.get('premise_text', 'Text not available'),
            hypothesis_text=sample.get('hypothesis_text', 'Text not available'),
            
            # Point cloud info
            total_points=stats['combined_total_points'],
            premise_tokens=premise_tokens.shape[0],
            hypothesis_tokens=hypothesis_tokens.shape[0],
            
            # Energy info
            forward_energy=model_analysis['combined']['forward_energy'],
            backward_energy=model_analysis['combined']['backward_energy'],
            asymmetric_energy=model_analysis['asymmetry_model']['asymmetric_energy'],
            
            # Topological features
            h0_features=len(h0_finite),
            h1_features=len(h1_finite),
            h0_max_persistence=np.max(h0_lifespans) if len(h0_lifespans) > 0 else 0.0,
            h1_max_persistence=np.max(h1_lifespans) if len(h1_lifespans) > 0 else 0.0,
            h0_total_persistence=np.sum(h0_lifespans) if len(h0_lifespans) > 0 else 0.0,
            h1_total_persistence=np.sum(h1_lifespans) if len(h1_lifespans) > 0 else 0.0,
            
            # Raw diagrams for plotting
            persistence_diagrams=diagrams
        )
        
        print(f"  H0 features: {analysis.h0_features} (total persistence: {analysis.h0_total_persistence:.4f})")
        print(f"  H1 features: {analysis.h1_features} (total persistence: {analysis.h1_total_persistence:.4f})")
        
        return analysis
    
    def plot_persistence_diagram(self, analysis: TopologicalAnalysis, save_path: Optional[str] = None):
        """
        Create a persistence diagram plot for one sample
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        h0_diagram = analysis.persistence_diagrams[0]
        h1_diagram = analysis.persistence_diagrams[1]
        
        # Filter finite persistence
        h0_finite = h0_diagram[np.isfinite(h0_diagram).all(axis=1)]
        h1_finite = h1_diagram[np.isfinite(h1_diagram).all(axis=1)]
        
        # Plot H0 (connected components)
        if len(h0_finite) > 0:
            ax1.scatter(h0_finite[:, 0], h0_finite[:, 1], c='blue', alpha=0.7, s=30)
            
            # Diagonal line
            max_val = max(h0_finite.max(), h1_finite.max()) if len(h1_finite) > 0 else h0_finite.max()
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax1.set_xlabel("Birth")
            ax1.set_ylabel("Death") 
            ax1.set_title(f"H0 Features (n={len(h0_finite)})")
            ax1.grid(True, alpha=0.3)
        
        # Plot H1 (loops/holes)
        if len(h1_finite) > 0:
            ax2.scatter(h1_finite[:, 0], h1_finite[:, 1], c='red', alpha=0.7, s=30)
            
            # Diagonal line
            max_val = max(h0_finite.max() if len(h0_finite) > 0 else 0, h1_finite.max())
            ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax2.set_xlabel("Birth")
            ax2.set_ylabel("Death")
            ax2.set_title(f"H1 Features (n={len(h1_finite)})")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No H1 features', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("H1 Features (n=0)")
        
        # Overall title
        title = f"{analysis.class_name.title()} - {analysis.dataset}\n"
        title += f"Energy: {analysis.forward_energy:.3f}, Points: {analysis.total_points}"
        fig.suptitle(title, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Persistence diagram saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_class_topologies(self, analyses: List[TopologicalAnalysis], save_dir: str):
        """
        Create comparative analysis plots across classes
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Group by class
        class_groups = {'entailment': [], 'neutral': [], 'contradiction': []}
        for analysis in analyses:
            class_groups[analysis.class_name].append(analysis)
        
        # Create comparison plots
        self._plot_energy_comparison(class_groups, f"{save_dir}/energy_comparison.png")
        self._plot_topology_comparison(class_groups, f"{save_dir}/topology_comparison.png")
        self._create_summary_table(class_groups, f"{save_dir}/topology_summary.txt")
    
    def _plot_energy_comparison(self, class_groups: Dict, save_path: str):
        """Plot energy distributions across classes"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = {'entailment': '#27AE60', 'neutral': '#F39C12', 'contradiction': '#E74C3C'}
        
        energy_types = ['forward_energy', 'backward_energy', 'asymmetric_energy']
        energy_labels = ['Forward Energy', 'Backward Energy', 'Asymmetric Energy']
        
        for i, (energy_type, label) in enumerate(zip(energy_types, energy_labels)):
            for class_name, analyses in class_groups.items():
                if analyses:
                    energies = [getattr(a, energy_type) for a in analyses]
                    axes[i].hist(energies, alpha=0.6, label=f"{class_name} (n={len(energies)})", 
                               color=colors[class_name], bins=10)
            
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Count")
            axes[i].set_title(f"{label} Distribution")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle("Energy Distributions by Class", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_topology_comparison(self, class_groups: Dict, save_path: str):
        """Plot topological feature distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = {'entailment': '#27AE60', 'neutral': '#F39C12', 'contradiction': '#E74C3C'}
        
        # H0 features
        for class_name, analyses in class_groups.items():
            if analyses:
                h0_counts = [a.h0_features for a in analyses]
                axes[0, 0].hist(h0_counts, alpha=0.6, label=f"{class_name}", 
                              color=colors[class_name], bins=10)
        
        axes[0, 0].set_xlabel("H0 Feature Count")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Connected Components (H0)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # H1 features
        for class_name, analyses in class_groups.items():
            if analyses:
                h1_counts = [a.h1_features for a in analyses]
                axes[0, 1].hist(h1_counts, alpha=0.6, label=f"{class_name}", 
                              color=colors[class_name], bins=10)
        
        axes[0, 1].set_xlabel("H1 Feature Count")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Loops/Holes (H1)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # H0 total persistence
        for class_name, analyses in class_groups.items():
            if analyses:
                h0_persistence = [a.h0_total_persistence for a in analyses]
                axes[1, 0].hist(h0_persistence, alpha=0.6, label=f"{class_name}", 
                              color=colors[class_name], bins=10)
        
        axes[1, 0].set_xlabel("Total H0 Persistence")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("H0 Persistence Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # H1 total persistence
        for class_name, analyses in class_groups.items():
            if analyses:
                h1_persistence = [a.h1_total_persistence for a in analyses]
                axes[1, 1].hist(h1_persistence, alpha=0.6, label=f"{class_name}", 
                              color=colors[class_name], bins=10)
        
        axes[1, 1].set_xlabel("Total H1 Persistence")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("H1 Persistence Distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle("Topological Feature Distributions by Class", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_summary_table(self, class_groups: Dict, save_path: str):
        """Create a summary table of topological statistics"""
        
        with open(save_path, 'w') as f:
            f.write("TOPOLOGICAL ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for class_name, analyses in class_groups.items():
                if not analyses:
                    continue
                
                f.write(f"{class_name.upper()} STATISTICS (n={len(analyses)})\n")
                f.write("-" * 30 + "\n")
                
                # Energy statistics
                forward_energies = [a.forward_energy for a in analyses]
                backward_energies = [a.backward_energy for a in analyses]
                asymmetric_energies = [a.asymmetric_energy for a in analyses]
                
                f.write("Energy Statistics:\n")
                f.write(f"  Forward:    {np.mean(forward_energies):.3f} ± {np.std(forward_energies):.3f}\n")
                f.write(f"  Backward:   {np.mean(backward_energies):.3f} ± {np.std(backward_energies):.3f}\n")
                f.write(f"  Asymmetric: {np.mean(asymmetric_energies):.3f} ± {np.std(asymmetric_energies):.3f}\n")
                
                # Topological statistics
                h0_features = [a.h0_features for a in analyses]
                h1_features = [a.h1_features for a in analyses]
                h0_persistence = [a.h0_total_persistence for a in analyses]
                h1_persistence = [a.h1_total_persistence for a in analyses]
                
                f.write("\nTopological Statistics:\n")
                f.write(f"  H0 Features:     {np.mean(h0_features):.1f} ± {np.std(h0_features):.1f}\n")
                f.write(f"  H1 Features:     {np.mean(h1_features):.1f} ± {np.std(h1_features):.1f}\n")
                f.write(f"  H0 Persistence:  {np.mean(h0_persistence):.3f} ± {np.std(h0_persistence):.3f}\n")
                f.write(f"  H1 Persistence:  {np.mean(h1_persistence):.3f} ± {np.std(h1_persistence):.3f}\n")
                
                # Point cloud statistics
                point_counts = [a.total_points for a in analyses]
                premise_tokens = [a.premise_tokens for a in analyses]
                hypothesis_tokens = [a.hypothesis_tokens for a in analyses]
                
                f.write(f"\nPoint Cloud Statistics:\n")
                f.write(f"  Total Points:    {np.mean(point_counts):.0f} ± {np.std(point_counts):.0f}\n")
                f.write(f"  Premise Tokens:  {np.mean(premise_tokens):.0f} ± {np.std(premise_tokens):.0f}\n")
                f.write(f"  Hypothesis Tokens: {np.mean(hypothesis_tokens):.0f} ± {np.std(hypothesis_tokens):.0f}\n")
                
                f.write("\n")
        
        print(f"Summary table saved: {save_path}")
    
    def find_representative_examples(self, analyses: List[TopologicalAnalysis], n_per_class: int = 3) -> List[TopologicalAnalysis]:
        """
        Find representative examples of each class based on topological characteristics
        """
        # Group by class
        class_groups = {'entailment': [], 'neutral': [], 'contradiction': []}
        for analysis in analyses:
            class_groups[analysis.class_name].append(analysis)
        
        representatives = []
        
        for class_name, class_analyses in class_groups.items():
            if not class_analyses:
                continue
            
            print(f"\nFinding {n_per_class} representative {class_name} examples:")
            
            if class_name == 'entailment':
                # For entailment: low forward energy, high H1 persistence (structured)
                scored = [(a, a.forward_energy * -1 + a.h1_total_persistence) for a in class_analyses]
            elif class_name == 'neutral':
                # For neutral: moderate energies, moderate topology
                mean_forward = np.mean([a.forward_energy for a in class_analyses])
                scored = [(a, -abs(a.forward_energy - mean_forward) + a.asymmetric_energy) for a in class_analyses]
            else:  # contradiction
                # For contradiction: high forward energy, many H1 features (complex)
                scored = [(a, a.forward_energy + a.h1_features * 0.1 + a.h1_total_persistence) for a in class_analyses]
            
            # Sort by score and take top N
            scored.sort(key=lambda x: x[1], reverse=True)
            top_examples = [analysis for analysis, score in scored[:n_per_class]]
            
            for i, example in enumerate(top_examples):
                print(f"  {i+1}. Energy: {example.forward_energy:.3f}, "
                      f"H1: {example.h1_features}, "
                      f"H1 pers: {example.h1_total_persistence:.3f}")
            
            representatives.extend(top_examples)
        
        return representatives
    
    def run_analysis(self, snli_path: str, mnli_path: str, 
                    samples_per_class: int = 20, 
                    save_dir: str = "topology_analysis") -> List[TopologicalAnalysis]:
        """
        Run complete topological analysis on both datasets
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("INDIVIDUAL SAMPLE TOPOLOGICAL ANALYSIS")
        print("="*80)
        
        all_analyses = []
        
        # Analyze SNLI samples
        if Path(snli_path).exists():
            print(f"\nAnalyzing SNLI samples...")
            snli_samples = self.load_dataset_samples("SNLI", snli_path, samples_per_class)
            
            for class_name, samples in snli_samples.items():
                for i, sample in enumerate(samples):
                    sample_id = f"SNLI_{class_name}_{i+1}"
                    analysis = self.analyze_sample_topology(sample, "SNLI", sample_id)
                    all_analyses.append(analysis)
                    
                    # Create individual persistence diagram for first few samples
                    if i < 3:
                        diagram_path = f"{save_dir}/persistence_diagram_{sample_id}.png"
                        self.plot_persistence_diagram(analysis, diagram_path)
        
        # Analyze MNLI samples
        if mnli_path is not None and Path(mnli_path).exists():
            print(f"\nAnalyzing MNLI samples...")
            mnli_samples = self.load_dataset_samples("MNLI", mnli_path, samples_per_class)
            
            for class_name, samples in mnli_samples.items():
                for i, sample in enumerate(samples):
                    sample_id = f"MNLI_{class_name}_{i+1}"
                    analysis = self.analyze_sample_topology(sample, "MNLI", sample_id)
                    all_analyses.append(analysis)
                    
                    # Create individual persistence diagram for first few samples
                    if i < 3:
                        diagram_path = f"{save_dir}/persistence_diagram_{sample_id}.png"
                        self.plot_persistence_diagram(analysis, diagram_path)
        
        # Create comparative analysis
        print(f"\nCreating comparative analysis...")
        self.compare_class_topologies(all_analyses, save_dir)
        
        # Find and plot representative examples
        representatives = self.find_representative_examples(all_analyses, n_per_class=3)
        
        print(f"\nCreating representative example diagrams...")
        for rep in representatives:
            rep_path = f"{save_dir}/representative_{rep.sample_id}.png"
            self.plot_persistence_diagram(rep, rep_path)
        
        print(f"\n" + "="*80)
        print("TOPOLOGY ANALYSIS COMPLETED")
        print(f"Total samples analyzed: {len(all_analyses)}")
        print(f"Representative examples: {len(representatives)}")
        print(f"Results saved in: {save_dir}")
        print("="*80)
        
        return all_analyses


def main():
    """
    Run individual sample topological analysis
    """
    
    # Model paths - update these to your trained models
    order_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/order_embedding_model_separate_margins.pt"
    asymmetry_model_path = "MSc_Topology_Codebase/phd_method/models/separate_models/new_independent_asymmetry_transform_model_v2.pt"
    
    # Data paths - update these to your datasets
    snli_data_path = "/vol/bitbucket/ahb24/tda_entailment_new/snli_val_sbert_tokens.pkl"
    mnli_data_path = None
    
    analyzer = IndividualTopologyAnalyzer(
        order_model_path=order_model_path,
        asymmetry_model_path=asymmetry_model_path,
        device='cuda'
    )
    
    # Run analysis
    results = analyzer.run_analysis(
        snli_path=snli_data_path,
        mnli_path=mnli_data_path,
        samples_per_class=20,
        save_dir="MSc_Topology_Codebase/phd_method/individual_topology_analysis"
    )
    
    print("Analysis completed! Check the individual_topology_analysis/ directory for results.")


if __name__ == "__main__":
    main()