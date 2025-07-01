import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json

"""
if 3-way margin loss performs better; add that to this file 
"""


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class PureHyperbolicLearnableOrderEmbeddingModel(nn.Module):
    """
    Pure Hyperbolic Order Embedding Model with learnable energy function parameters
    
    This model:
    1. Uses only hyperbolic operations throughout (pure hyperbolic)
    2. Learns optimal energy function parameters during training
    3. Uses same parameters for all classes (blind testing compatible)
    4. Should give maximum theoretical performance benefits
    """
    
    def __init__(self, bert_dim: int = 768, order_dim: int = 50, asymmetry_weight: float = 0.2):
        super().__init__()
        self.bert_dim = bert_dim
        self.order_dim = order_dim
        self.asymmetry_weight = asymmetry_weight
        
        # Poincaré ball manifold
        self.ball = geoopt.PoincareBall()
        
        # PURE HYPERBOLIC NEURAL NETWORK LAYERS
        # All parameters live on the hyperbolic manifold
        
        # Initial projection dimension (we need to map from 768D BERT to smaller space first)
        intermediate_dim = min(order_dim * 2, 100)  # Reasonable intermediate size
        
        # Hyperbolic weight matrices - these are ManifoldParameters
        self.hyp_weight1 = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(bert_dim, intermediate_dim),
            manifold=self.ball
        )
        self.hyp_bias1 = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(intermediate_dim),
            manifold=self.ball
        )
        
        self.hyp_weight2 = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(intermediate_dim, order_dim),
            manifold=self.ball
        )
        self.hyp_bias2 = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(order_dim),
            manifold=self.ball
        )
        
        # Asymmetric projection weights (also hyperbolic)
        self.asym_weight = geoopt.ManifoldParameter(
            self._init_hyperbolic_matrix(order_dim, order_dim),
            manifold=self.ball
        )
        self.asym_bias = geoopt.ManifoldParameter(
            self._init_hyperbolic_vector(order_dim),
            manifold=self.ball
        )
        
        # LEARNABLE ENERGY FUNCTION PARAMETERS
        # These are standard parameters (not manifold parameters)
        # because they're scalars, not points in hyperbolic space
        
        self.specificity_scaling = nn.Parameter(
            torch.tensor(0.5),  # Initialize with reasonable guess
            requires_grad=True
        )
        self.base_tolerance = nn.Parameter(
            torch.tensor(0.3),  # Initialize with reasonable guess
            requires_grad=True
        )
        self.proximity_weight = nn.Parameter(
            torch.tensor(0.3),  # Weight for proximity vs specificity violation
            requires_grad=True
        )
        
        # Additional learnable parameters for energy function refinement
        self.specificity_power = nn.Parameter(
            torch.tensor(1.0),  # Power for specificity difference (allows non-linear scaling)
            requires_grad=True
        )
        self.distance_scaling = nn.Parameter(
            torch.tensor(1.0),  # Overall scaling for hyperbolic distances
            requires_grad=True
        )
        
        # Parameter constraints (buffers - not learnable)
        self.register_buffer('min_scaling', torch.tensor(0.05))
        self.register_buffer('max_scaling', torch.tensor(3.0))
        self.register_buffer('min_tolerance', torch.tensor(0.01))
        self.register_buffer('max_tolerance', torch.tensor(1.5))
        self.register_buffer('min_weight', torch.tensor(0.01))
        self.register_buffer('max_weight', torch.tensor(2.0))
        self.register_buffer('min_power', torch.tensor(0.5))
        self.register_buffer('max_power', torch.tensor(2.0))
        
        # Numerical stability
        self.eps = 1e-8
        self.max_norm = 0.95  # Stay away from boundary of unit ball


    def _init_hyperbolic_matrix(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Initialize hyperbolic weight matrix with small values"""
        # Start with small random values and ensure they're in the unit ball
        matrix = torch.randn(in_dim, out_dim) * 0.01
        # Ensure all points are inside unit ball
        norms = torch.norm(matrix, dim=-1, keepdim=True)
        matrix = torch.where(norms >= self.max_norm, matrix * (self.max_norm / (norms + self.eps)), matrix)
        return matrix

    def _init_hyperbolic_vector(self, dim: int) -> torch.Tensor:
        """Initialize hyperbolic bias vector"""
        vector = torch.randn(dim) * 0.01
        norm = torch.norm(vector)
        if norm >= self.max_norm:
            vector = vector * (self.max_norm / (norm + self.eps))
        return vector

    def get_constrained_parameters(self):
        """Get energy function parameters constrained to reasonable ranges"""
        specificity_scaling = torch.clamp(self.specificity_scaling, self.min_scaling, self.max_scaling)
        base_tolerance = torch.clamp(self.base_tolerance, self.min_tolerance, self.max_tolerance)
        proximity_weight = torch.clamp(self.proximity_weight, self.min_weight, self.max_weight)
        specificity_power = torch.clamp(self.specificity_power, self.min_power, self.max_power)
        distance_scaling = torch.clamp(self.distance_scaling, self.min_scaling, self.max_scaling)
        
        return specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling

    def mobius_linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Möbius linear transformation - the pure hyperbolic equivalent of nn.Linear
        
        This implements the mathematical operation for linear transformations
        in hyperbolic space using the Möbius gyrovector space operations.
        
        Based on: "Hyperbolic Neural Networks" (Ganea et al., 2018)
        """
        batch_size = x.shape[0]
        
        # For numerical stability, we work in the tangent space at origin
        # This is mathematically correct for the Poincaré ball model
        
        # Step 1: Map input to tangent space at origin
        x_tangent = self.ball.logmap0(x)
        
        # Step 2: Apply Euclidean transformation in tangent space
        # Note: This is correct! Tangent spaces are Euclidean
        if weight.dim() == 2:
            # Matrix multiplication for weight matrix
            if x_tangent.shape[-1] != weight.shape[0]:
                # Handle dimension mismatch by projecting
                proj_size = min(x_tangent.shape[-1], weight.shape[0])
                x_tangent = x_tangent[..., :proj_size]
                weight_used = weight[:proj_size, :]
            else:
                weight_used = weight
            
            transformed = torch.matmul(x_tangent, weight_used)
        else:
            # Element-wise for bias vector
            transformed = x_tangent + bias[:x_tangent.shape[-1]]
        
        # Step 3: Map back to hyperbolic space with scaling for stability
        scale_factor = 0.5  # Conservative scaling to stay inside unit ball
        result = self.ball.expmap0(transformed * scale_factor)
        
        # Step 4: Ensure points stay inside unit ball
        norms = torch.norm(result, dim=-1, keepdim=True)
        result = torch.where(norms >= self.max_norm, result * (self.max_norm / (norms + self.eps)), result)
        
        return result

    def hyperbolic_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic nonlinearity function
        
        We map to tangent space, apply nonlinearity, then map back.
        Uses tanh which works well in hyperbolic space.
        """
        # Map to tangent space
        x_tangent = self.ball.logmap0(x)
        
        # Apply nonlinearity in tangent space
        # Tanh is preferred over ReLU in hyperbolic space
        activated = torch.tanh(x_tangent)
        
        # Map back with conservative scaling
        result = self.ball.expmap0(activated * 0.3)
        
        # Ensure inside unit ball
        norms = torch.norm(result, dim=-1, keepdim=True)
        result = torch.where(norms >= self.max_norm, result * (self.max_norm / (norms + self.eps)), result)
        
        return result

    def forward(self, bert_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pure hyperbolic forward pass
        
        Args:
            bert_embeddings: [batch_size, bert_dim] BERT embeddings (Euclidean)
            
        Returns:
            hyperbolic_embeddings: [batch_size, order_dim] pure hyperbolic embeddings
        """
        batch_size = bert_embeddings.shape[0]
        
        # Initial mapping from Euclidean BERT space to hyperbolic space
        # This is unavoidable since BERT embeddings are Euclidean
        initial_scale = 0.001  # Very conservative initial scaling
        
        # Take subset of BERT dimensions to fit our intermediate dimension
        input_subset = bert_embeddings[:, :self.hyp_weight1.shape[0]]
        
        # Initial projection to hyperbolic space
        x = self.ball.expmap0(input_subset * initial_scale)
        
        # First pure hyperbolic layer
        x = self.mobius_linear(x, self.hyp_weight1, self.hyp_bias1)
        x = self.hyperbolic_nonlinearity(x)
        
        # Second pure hyperbolic layer  
        x = self.mobius_linear(x, self.hyp_weight2, self.hyp_bias2)
        x = self.hyperbolic_nonlinearity(x)
        
        return x

    def learnable_hyperbolic_order_violation_energy(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Pure hyperbolic order violation energy with learnable parameters
        
        This uses only hyperbolic operations and learns optimal parameters
        for the energy function during training.
        """
        # Get constrained learnable parameters
        specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling = self.get_constrained_parameters()
        
        # Component 1: Specificity violation (pure hyperbolic)
        u_norms = self.ball.dist0(u_emb) * distance_scaling  # Distance from origin
        v_norms = self.ball.dist0(v_emb) * distance_scaling
        
        # For entailment: v should be more specific than u (v_norm > u_norm)
        specificity_violation = torch.relu(u_norms - v_norms)
        
        # Component 2: Learnable proximity violation
        pairwise_dist = self.ball.dist(u_emb, v_emb) * distance_scaling
        specificity_diff = torch.abs(u_norms - v_norms)
        
        # LEARNABLE expected geodesic formula with power scaling
        expected_geodesic = (specificity_diff ** specificity_power) * specificity_scaling + base_tolerance
        proximity_violation = torch.relu(pairwise_dist - expected_geodesic)
        
        # Learnable combination of violations
        total_violation = specificity_violation + proximity_weight * proximity_violation
        
        return {
            'total_energy': total_violation,
            'specificity_violation': specificity_violation,
            'proximity_violation': proximity_violation,
            'expected_geodesic': expected_geodesic,
            'u_norms': u_norms,
            'v_norms': v_norms,
            'pairwise_dist': pairwise_dist,
            'learned_params': {
                'specificity_scaling': specificity_scaling.item(),
                'base_tolerance': base_tolerance.item(),
                'proximity_weight': proximity_weight.item(),
                'specificity_power': specificity_power.item(),
                'distance_scaling': distance_scaling.item()
            }
        }

    def hyperbolic_asymmetric_features(self, hyperbolic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pure hyperbolic asymmetric feature transformation
        """
        return self.mobius_linear(hyperbolic_embeddings, self.asym_weight, self.asym_bias)

    def hyperbolic_asymmetric_energy(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> torch.Tensor:
        """
        Pure hyperbolic asymmetric energy using only hyperbolic distances
        """
        premise_asym = self.hyperbolic_asymmetric_features(premise_emb)
        hypothesis_asym = self.hyperbolic_asymmetric_features(hypothesis_emb)
        
        # Use pure hyperbolic distance
        _, _, _, _, distance_scaling = self.get_constrained_parameters()
        return self.ball.dist(premise_asym, hypothesis_asym) * distance_scaling

    def compute_bidirectional_energies(self, premise_emb: torch.Tensor, hypothesis_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all energies using pure hyperbolic operations and learnable parameters
        """
        # Forward and backward order violation energies
        forward_result = self.learnable_hyperbolic_order_violation_energy(premise_emb, hypothesis_emb)
        backward_result = self.learnable_hyperbolic_order_violation_energy(hypothesis_emb, premise_emb)
        
        # Pure hyperbolic asymmetric energy
        asym_energy = self.hyperbolic_asymmetric_energy(premise_emb, hypothesis_emb)
        
        # Asymmetry measure
        forward_energy = forward_result['total_energy']
        backward_energy = backward_result['total_energy']
        asymmetry_measure = torch.abs(forward_energy - backward_energy)
        
        return {
            'forward_energy': forward_energy,
            'backward_energy': backward_energy,
            'asymmetric_energy': asym_energy,
            'asymmetry_measure': asymmetry_measure,
            'learned_params': forward_result['learned_params'],
            'detailed_forward': forward_result,
            'detailed_backward': backward_result
        }

    def get_learned_parameters_summary(self) -> Dict[str, float]:
        """
        Get a summary of the current learned parameter values
        Useful for monitoring training and understanding what the model learned
        """
        specificity_scaling, base_tolerance, proximity_weight, specificity_power, distance_scaling = self.get_constrained_parameters()
        
        return {
            'specificity_scaling': specificity_scaling.item(),
            'base_tolerance': base_tolerance.item(),
            'proximity_weight': proximity_weight.item(),
            'specificity_power': specificity_power.item(),
            'distance_scaling': distance_scaling.item()
        }


class PureHyperbolicOrderEmbeddingTrainer:
    """
    Enhanced trainer with 3-way margin loss and corrected asymmetric loss
    """
    
    def __init__(self, model: PureHyperbolicLearnableOrderEmbeddingModel, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        
        # Use Riemannian optimizer for hyperbolic parameters
        self.optimizer = geoopt.optim.RiemannianAdam([
            {'params': [p for n, p in model.named_parameters() if 'hyp_' in n or 'asym_' in n], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if 'hyp_' not in n and 'asym_' not in n], 'lr': 5e-3}
        ], weight_decay=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.energy_rankings = []
        self.parameter_history = []


    def compute_enhanced_hyperbolic_loss(self, premise_embs: torch.Tensor, hypothesis_embs: torch.Tensor, 
                                       labels: torch.Tensor, label_strs: List[str], 
                                       margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Enhanced 3-way hyperbolic loss with corrected asymmetric patterns
        
        Args:
            premise_embs: Premise BERT embeddings
            hypothesis_embs: Hypothesis BERT embeddings
            labels: Label indices (0=entailment, 1=neutral, 2=contradiction)
            label_strs: String labels
            margin: Base margin for class separation
            
        Returns:
            Total loss, forward energies, and energy statistics
        """
        # Get pure hyperbolic embeddings
        premise_hyp = self.model(premise_embs)
        hypothesis_hyp = self.model(hypothesis_embs)
        
        # Compute hyperbolic order violation energies
        energy_dict = self.model.compute_bidirectional_energies(premise_hyp, hypothesis_hyp)
        forward_energies = energy_dict['forward_energy']
        
        # Create masks for each class
        entailment_mask = (labels == 0)
        neutral_mask = (labels == 1)
        contradiction_mask = (labels == 2)
        
        # 3-CLASS HYPERBOLIC MAX-MARGIN LOSS
        total_loss = 0.0
        
        # Entailment pairs: minimize energy (target ~0)
        if entailment_mask.any():
            entailment_energies = forward_energies[entailment_mask]
            entailment_loss = entailment_energies.mean()
            total_loss += entailment_loss
        
        # Neutral pairs: medium energy (target ~margin)
        if neutral_mask.any():
            neutral_energies = forward_energies[neutral_mask]
            # Loss if energy is too low (< margin/2) or too high (> 2*margin)
            neutral_loss = (
                torch.clamp(margin/2 - neutral_energies, min=0).mean() +  # Push up if too low
                torch.clamp(neutral_energies - 2*margin, min=0).mean()    # Push down if too high
            )
            total_loss += neutral_loss
        
        # Contradiction pairs: high energy (target ~2*margin)
        if contradiction_mask.any():
            contradiction_energies = forward_energies[contradiction_mask]
            # Loss if energy is too low (should be > 1.5*margin)
            contradiction_loss = torch.clamp(1.5*margin - contradiction_energies, min=0).mean()
            total_loss += contradiction_loss
        
        # RANKING LOSSES: Ensure E < N < C hierarchy
        ranking_loss = 0.0
        small_margin = 0.5
        
        # Entailment should have lower energy than neutral
        if entailment_mask.any() and neutral_mask.any():
            ent_mean = forward_energies[entailment_mask].mean()
            neut_mean = forward_energies[neutral_mask].mean()
            ranking_loss += torch.clamp(ent_mean - neut_mean + small_margin, min=0)
        
        # Neutral should have lower energy than contradiction
        if neutral_mask.any() and contradiction_mask.any():
            neut_mean = forward_energies[neutral_mask].mean()
            cont_mean = forward_energies[contradiction_mask].mean()
            ranking_loss += torch.clamp(neut_mean - cont_mean + small_margin, min=0)
        
        # Entailment should have lower energy than contradiction
        if entailment_mask.any() and contradiction_mask.any():
            ent_mean = forward_energies[entailment_mask].mean()
            cont_mean = forward_energies[contradiction_mask].mean()
            ranking_loss += torch.clamp(ent_mean - cont_mean + small_margin, min=0)
        
        # Combine main and ranking losses
        standard_loss = total_loss + 0.5 * ranking_loss
        
        # CORRECTED ASYMMETRIC LOSS
        asymmetric_loss = self.compute_corrected_asymmetric_loss(premise_hyp, hypothesis_hyp, labels, label_strs)
        
        # TOTAL COMBINED LOSS
        total_loss = standard_loss + self.model.asymmetry_weight * asymmetric_loss
        
        # Compute energy statistics for monitoring
        energy_stats = self._compute_energy_statistics(energy_dict, label_strs)
        
        return total_loss, forward_energies, energy_stats
    

    def compute_corrected_asymmetric_loss(self, premise_hyp: torch.Tensor, hypothesis_hyp: torch.Tensor, 
                                        labels: torch.Tensor, label_strs: List[str]) -> torch.Tensor:
        """
        Corrected asymmetric loss: Both entailment and contradiction have HIGH asymmetry
        
        Key insight:
        - Entailment: HIGH asymmetry with specific pattern (low forward, high backward)
        - Neutral: LOW asymmetry (symmetric - both directions unrelated)
        - Contradiction: HIGH asymmetry with different pattern (high forward, variable backward)
        """
        energy_dict = self.model.compute_bidirectional_energies(premise_hyp, hypothesis_hyp)
        
        asymmetric_loss = 0.0
        for i, label_str in enumerate(label_strs):
            forward_e = energy_dict['forward_energy'][i]
            backward_e = energy_dict['backward_energy'][i]
            asymmetric_e = energy_dict['asymmetric_energy'][i]
            asymmetry_measure = torch.abs(forward_e - backward_e)
            
            if label_str == 'entailment':
                # SPECIFIC asymmetric pattern: low forward, high backward
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, torch.tensor(0.1, device=self.device))
                asymmetric_loss += torch.nn.functional.mse_loss(backward_e, torch.tensor(0.8, device=self.device))
                # Encourage HIGH asymmetry magnitude
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.7, device=self.device))
                
            elif label_str == 'neutral':
                # LOW asymmetry (symmetric)
                target_energy = torch.tensor(0.6, device=self.device)
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, target_energy)
                asymmetric_loss += torch.nn.functional.mse_loss(backward_e, target_energy)
                # Encourage LOW asymmetry magnitude - key insight!
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.0, device=self.device))
                
            elif label_str == 'contradiction':
                # DIFFERENT asymmetric pattern: high forward, variable backward
                asymmetric_loss += torch.nn.functional.mse_loss(forward_e, torch.tensor(0.9, device=self.device))
                # Allow variable backward energy (contradiction can have different patterns)
                
                # Encourage HIGH asymmetry magnitude (similar to entailment)
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetry_measure, torch.tensor(0.6, device=self.device))
                
                # Also encourage high asymmetric energy
                asymmetric_loss += torch.nn.functional.mse_loss(asymmetric_e, torch.tensor(0.8, device=self.device))
        
        return asymmetric_loss / len(label_strs)


    def _compute_energy_statistics(self, energy_dict: Dict[str, torch.Tensor], 
                                 label_strs: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute energy statistics for monitoring"""
        stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        forward_energies = energy_dict['forward_energy']
        backward_energies = energy_dict['backward_energy']
        asymmetry_measures = energy_dict['asymmetry_measure']
        asymmetric_energies = energy_dict['asymmetric_energy']
        
        for i, label_str in enumerate(label_strs):
            if label_str in stats:
                stats[label_str].append({
                    'forward_energy': forward_energies[i].item(),
                    'backward_energy': backward_energies[i].item(),
                    'asymmetry_measure': asymmetry_measures[i].item(),
                    'asymmetric_energy': asymmetric_energies[i].item()
                })
        
        # Compute summary statistics
        summary_stats = {}
        for label, energy_list in stats.items():
            if energy_list:
                forward_energies = [e['forward_energy'] for e in energy_list]
                backward_energies = [e['backward_energy'] for e in energy_list]
                asymmetries = [e['asymmetry_measure'] for e in energy_list]
                asym_energies = [e['asymmetric_energy'] for e in energy_list]
                
                summary_stats[label] = {
                    'mean_forward_energy': np.mean(forward_energies),
                    'std_forward_energy': np.std(forward_energies),
                    'mean_backward_energy': np.mean(backward_energies),
                    'std_backward_energy': np.std(backward_energies),
                    'mean_asymmetry': np.mean(asymmetries),
                    'std_asymmetry': np.std(asymmetries),
                    'mean_asymmetric_energy': np.mean(asym_energies),
                    'std_asymmetric_energy': np.std(asym_energies)
                }
        
        return summary_stats

    def train_epoch(self, dataloader) -> float:
        """Train one epoch with enhanced 3-way loss and corrected asymmetric patterns"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Track parameter values during training
        epoch_params = []
        
        for batch in dataloader:
            premise_embs = batch['premise_emb'].to(self.device)
            hypothesis_embs = batch['hypothesis_emb'].to(self.device)
            labels = batch['label'].to(self.device)
            label_strs = batch['label_str']
            
            self.optimizer.zero_grad()
            
            # Compute enhanced 3-way hyperbolic loss
            loss, forward_energies, energy_stats = self.compute_enhanced_hyperbolic_loss(
                premise_embs, hypothesis_embs, labels, label_strs
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Track learned parameters
            if num_batches % 10 == 0:  # Every 10 batches
                params = self.model.get_learned_parameters_summary()
                epoch_params.append(params)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Store parameter evolution
        if epoch_params:
            avg_params = {k: np.mean([p[k] for p in epoch_params]) for k in epoch_params[0].keys()}
            self.parameter_history.append(avg_params)
        
        return avg_loss
    
    def evaluate(self, dataloader) -> Tuple[float, Dict]:
        """Evaluate model with enhanced 3-way loss and energy analysis"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Collect energies by label for detailed analysis
        all_stats = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        with torch.no_grad():
            for batch in dataloader:
                premise_embs = batch['premise_emb'].to(self.device)
                hypothesis_embs = batch['hypothesis_emb'].to(self.device)
                labels = batch['label'].to(self.device)
                label_strs = batch['label_str']
                
                loss, forward_energies, energy_stats = self.compute_enhanced_hyperbolic_loss(
                    premise_embs, hypothesis_embs, labels, label_strs
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Accumulate energy statistics
                for label in all_stats:
                    if label in energy_stats:
                        all_stats[label].extend(energy_stats[label])
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Compute comprehensive summary statistics
        summary_stats = {}
        for label, energy_list in all_stats.items():
            if energy_list:
                # Extract all energy types
                forward_energies = [e['forward_energy'] for e in energy_list]
                backward_energies = [e['backward_energy'] for e in energy_list]
                asymmetries = [e['asymmetry_measure'] for e in energy_list]
                asym_energies = [e['asymmetric_energy'] for e in energy_list]
                
                summary_stats[label] = {
                    'count': len(energy_list),
                    'forward_energy': {
                        'mean': np.mean(forward_energies),
                        'std': np.std(forward_energies)
                    },
                    'backward_energy': {
                        'mean': np.mean(backward_energies),
                        'std': np.std(backward_energies)
                    },
                    'asymmetry_measure': {
                        'mean': np.mean(asymmetries),
                        'std': np.std(asymmetries)
                    },
                    'asymmetric_energy': {
                        'mean': np.mean(asym_energies),
                        'std': np.std(asym_energies)
                    }
                }
        
        self.energy_rankings.append(summary_stats)
        
        return avg_loss, summary_stats


    
if __name__ == "__main__":

    model = PureHyperbolicLearnableOrderEmbeddingModel()
    
    # Show initial learned parameters
    initial_params = model.get_learned_parameters_summary()
    print(f"Initial learned parameters: {initial_params}")
    
    # Test forward pass
    bert_embs = torch.randn(5, 768)
    hyp_embs = model(bert_embs)
    print(f"Input shape: {bert_embs.shape}")
    print(f"Output shape: {hyp_embs.shape}")
    print(f"Output in unit ball: {torch.all(torch.norm(hyp_embs, dim=-1) < 1.0)}")
    
    # Test energy computation
    premise = hyp_embs[:2]
    hypothesis = hyp_embs[2:4]
    
    energies = model.compute_bidirectional_energies(premise, hypothesis)
    print(f"Forward energies: {energies['forward_energy']}")
    print(f"Learned parameters: {energies['learned_params']}")
    
    print(f"\n✅ Pure hyperbolic model ready for training!")
