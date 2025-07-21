import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Dict

try:
    from torch_topological.nn import VietorisRipsComplex, WassersteinDistance
    # Import the PersistenceInformation class to use for type checking and wrapping
    from torch_topological.nn.data import PersistenceInformation
    TORCH_TOPOLOGICAL_AVAILABLE = True
    print("torch-topological successfully imported")
except ImportError as e:
    TORCH_TOPOLOGICAL_AVAILABLE = False
    print(f"WARNING: torch-topological not available. Error: {e}")

# FIX 1: Create a simple wrapper class for our pre-computed target diagrams.
# This gives them the `.diagram` attribute that WassersteinDistance expects.
class TargetDiagramWrapper:
    def __init__(self, diagram_tensor):
        self.diagram = diagram_tensor

class TorchTopologicalPersistenceLoss(nn.Module):
    """
    Topological persistence loss using the torch-topological library.
    This version correctly handles the library's data structures.
    """
    
    def __init__(self, prototypes_path: str, min_samples_per_class: int = 200, 
        loss_scale_factor: float = 1.0, max_dimension: int = 1,
        max_points_per_class: int = 500):
        
        super().__init__()
        
        if not TORCH_TOPOLOGICAL_AVAILABLE:
            raise ImportError("torch-topological is required for this loss function.")
        
        with open(prototypes_path, 'rb') as f:
            self.prototypes = pickle.load(f)
        
        self.min_samples_per_class = min_samples_per_class
        self.max_points_per_class = max_points_per_class
        self.class_names = ['entailment', 'neutral', 'contradiction']
        self.loss_scale_factor = loss_scale_factor
        
        self.vr_complex = VietorisRipsComplex(dim=max_dimension)
        self.wasserstein_distance = WassersteinDistance(q=2)
        
        self._preprocess_target_diagrams()
        
        print("TorchTopologicalPersistenceLoss Initialized Correctly.")

    def _preprocess_target_diagrams(self):
        """
        Load target diagrams and convert them from numpy arrays to our
        simple TargetDiagramWrapper.
        """
        self.target_diagrams = {}
        print("Pre-processing target diagrams and converting to tensors...")
        
        for class_name, class_prototypes in self.prototypes.items():
            if class_name in self.class_names:
                target_diagram_np = class_prototypes.get('H1', np.array([]))
                
                if target_diagram_np.size > 0:
                    target_tensor = torch.from_numpy(target_diagram_np).float()
                else:
                    target_tensor = torch.empty((0, 2), dtype=torch.float32)
                
                # Store the wrapped tensor.
                self.target_diagrams[class_name] = TargetDiagramWrapper(target_tensor)
                print(f"  {class_name}: shape {target_tensor.shape}")

                #DEBUG
                # if target_tensor.shape[0] > 0:
                #     total_persistence = (target_tensor[:, 1] - target_tensor[:, 0]).sum().item()
                #     print(f"    Total persistence: {total_persistence:.4f}")
                #     print(f"    Feature range: birth[{target_tensor[:, 0].min():.4f}, {target_tensor[:, 0].max():.4f}]")
                #     print(f"                   death[{target_tensor[:, 1].min():.4f}, {target_tensor[:, 1].max():.4f}]")

    def _compute_persistence_information(self, features: torch.Tensor):
        """
        Compute persistence and return the RAW PersistenceInformation object.
        DO NOT extract the tensor here.
        """
        
        #DEBUG
        # print(f"      Input features shape: {features.shape}")
        # print(f"      Features device: {features.device}")
        # print(f"      Features dtype: {features.dtype}")
        # print(f"      Features range: [{features.min():.4f}, {features.max():.4f}]")
        # print(f"      Features mean: {features.mean():.4f}, std: {features.std():.4f}")

        if features.shape[0] > self.max_points_per_class:
            indices = torch.randperm(features.shape[0])[:self.max_points_per_class]
            features = features[indices]

        try:
            # print(f"      Calling VietorisRipsComplex...")
            persistence_info_list = self.vr_complex(features)
            # print(f"      VR returned {len(persistence_info_list)} dimensions")
            
            # Return the PersistenceInformation object for H1, or None if it doesn't exist
            if len(persistence_info_list) > 1 and persistence_info_list[1] is not None:
                
                #DEBUG
                # h1_pi = persistence_info_list[1]
                # print(f"      H1 PersistenceInformation type: {type(h1_pi)}")
                # if hasattr(h1_pi, 'diagram'):
                #     diagram = h1_pi.diagram
                #     print(f"      H1 diagram shape: {diagram.shape}")
                #     print(f"      H1 diagram type: {type(diagram)}")
                #     if diagram.shape[0] > 0:
                #         if isinstance(diagram, np.ndarray):
                #             persistence_sum = np.sum(diagram[:, 1] - diagram[:, 0])
                #         else:
                #             persistence_sum = torch.sum(diagram[:, 1] - diagram[:, 0]).item()
                #             print(f"      H1 total persistence: {persistence_sum:.6f}")
                #     else:
                #         print(f"      H1 diagram is EMPTY")



                return persistence_info_list[1]
            else:
                print(f"      No H1 dimension found")
                return None
            
        except Exception as e:
            print(f"    WARNING: Persistence computation failed: {e}")
            return None

    # def forward(self, latent_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute topological loss by passing the correct objects to the loss function.
    #     """
    #     total_loss = torch.tensor(0.0, device=latent_features.device, requires_grad=True)
    #     valid_classes = 0
    #     device = latent_features.device
        
    #     TOP_K_FEATURES = 100
        
    #     for class_idx, class_name in enumerate(self.class_names):
    #         class_mask = (labels == class_idx)
            
    #         if class_mask.sum().item() < self.min_samples_per_class:
    #             continue
                
    #         class_features = latent_features[class_mask]
            
    #         try:
    #             # FIX 2: Get the full PersistenceInformation object for the current batch.
    #             current_pi = self._compute_persistence_information(class_features)
    #             if current_pi is None:
    #                 print(f"      Skipping {class_name}: persistence computation failed")
    #                 continue

    #             # Get the wrapped target diagram and move its inner tensor to the correct device.
    #             target_wrapped = self.target_diagrams[class_name]
    #             target_wrapped.diagram = target_wrapped.diagram.to(device)

    #             # --- NEW: PERSISTENCE FILTERING LOGIC ---

    #             # A. Filter the CURRENT diagram from the model
    #             current_diagram_tensor = current_pi.diagram
    #             if current_diagram_tensor.shape[0] > TOP_K_FEATURES:
    #                 persistences = current_diagram_tensor[:, 1] - current_diagram_tensor[:, 0]
    #                 top_indices = torch.topk(persistences, k=TOP_K_FEATURES).indices
    #                 filtered_current_diagram = current_diagram_tensor[top_indices]
    #             else:
    #                 filtered_current_diagram = current_diagram_tensor

    #             # B. Filter the TARGET diagram
    #             target_diagram_tensor = target_wrapped.diagram
    #             if target_diagram_tensor.shape[0] > TOP_K_FEATURES:
    #                 persistences = target_diagram_tensor[:, 1] - target_diagram_tensor[:, 0]
    #                 top_indices = torch.topk(persistences, k=TOP_K_FEATURES).indices
    #                 filtered_target_diagram = target_diagram_tensor[top_indices]
    #             else:
    #                 filtered_target_diagram = target_diagram_tensor

    #             # C. Wrap the new, smaller tensors for the loss function
    #             filtered_current_pi = TargetDiagramWrapper(filtered_current_diagram)
    #             filtered_target_pi = TargetDiagramWrapper(filtered_target_diagram)

    #             # --- END OF NEW LOGIC ---
                
    #             # FIX 3: Call WassersteinDistance with the objects the docs expect.
    #             # Passing them as single-item lists is the most robust method.
    #             # print(f"      Calling WassersteinDistance...")

    #             class_loss = self.wasserstein_distance([filtered_current_pi], [filtered_target_pi])
    #             # print(f"      {class_name} raw loss: {class_loss}")
    #             # if hasattr(class_loss, 'mean'):
    #             #     print(f"      {class_name} mean loss: {class_loss.mean().item():.6f}")
                
    #             total_loss = total_loss + (class_loss.mean() * self.loss_scale_factor)
    #             valid_classes += 1
                
    #         except Exception as e:
    #             print(f"    WARNING: torch-topological loss calculation failed for {class_name}: {e}")
    #             continue
        
    #     if valid_classes > 0:
    #         return total_loss / valid_classes
    #     else:
    #         return torch.tensor(0.0, device=device, requires_grad=True)

    # def forward(self, latent_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute topological loss with detailed debugging print statements.
    #     """
    #     total_loss = torch.tensor(0.0, device=latent_features.device, requires_grad=True)
    #     valid_classes = 0
    #     device = latent_features.device
        
    #     TOP_K_FEATURES = 100

    #     print("\n--- NEW BATCH ---")
    #     for class_idx, class_name in enumerate(self.class_names):
    #         class_mask = (labels == class_idx)
            
    #         if class_mask.sum().item() < self.min_samples_per_class:
    #             continue
                
    #         class_features = latent_features[class_mask]
    #         print(f"\n[Class: {class_name.upper()}]")
    #         print(f"  1. Input Features Shape: {class_features.shape}")

    #         try:
    #             # 1. Get the full PersistenceInformation object
    #             current_pi = self._compute_persistence_information(class_features)
    #             if current_pi is None or not hasattr(current_pi, 'diagram'):
    #                 print("  2. FAILED to compute a valid persistence diagram.")
    #                 continue
                
    #             current_diagram_tensor = current_pi.diagram
    #             print(f"  2. Computed Diagram: {current_diagram_tensor.shape[0]} features")

    #             # 2. Get the wrapped target diagram
    #             target_wrapped = self.target_diagrams[class_name]
    #             target_diagram_tensor = target_wrapped.diagram.to(device)
    #             print(f"  3. Target Diagram: {target_diagram_tensor.shape[0]} features")
                
    #             # --- PERSISTENCE FILTERING & DIAGNOSTICS ---

    #             # A. Analyze and Filter the CURRENT diagram
    #             if current_diagram_tensor.shape[0] > 0:
    #                 current_persistences = current_diagram_tensor[:, 1] - current_diagram_tensor[:, 0]
    #                 print(f"  4a. CURRENT Diagram - Top 5 Persistences: {torch.topk(current_persistences, k=min(5, len(current_persistences))).values.detach().cpu().numpy()}")
                    
    #                 if current_diagram_tensor.shape[0] > TOP_K_FEATURES:
    #                     top_indices = torch.topk(current_persistences, k=TOP_K_FEATURES).indices
    #                     filtered_current_diagram = current_diagram_tensor[top_indices]
    #                 else:
    #                     filtered_current_diagram = current_diagram_tensor
    #             else:
    #                 filtered_current_diagram = current_diagram_tensor
                
    #             print(f"  4b. CURRENT Diagram - Shape After Filtering: {filtered_current_diagram.shape}")


    #             # B. Analyze and Filter the TARGET diagram
    #             if target_diagram_tensor.shape[0] > 0:
    #                 target_persistences = target_diagram_tensor[:, 1] - target_diagram_tensor[:, 0]
    #                 print(f"  5a. TARGET Diagram - Top 5 Persistences: {torch.topk(target_persistences, k=min(5, len(target_persistences))).values.detach().cpu().numpy()}")

    #                 if target_diagram_tensor.shape[0] > TOP_K_FEATURES:
    #                     top_indices = torch.topk(target_persistences, k=TOP_K_FEATURES).indices
    #                     filtered_target_diagram = target_diagram_tensor[top_indices]
    #                 else:
    #                     filtered_target_diagram = target_diagram_tensor
    #             else:
    #                 filtered_target_diagram = target_diagram_tensor

    #             print(f"  5b. TARGET Diagram - Shape After Filtering: {filtered_target_diagram.shape}")


    #             # C. Wrap the new, smaller tensors for the loss function
    #             filtered_current_pi = TargetDiagramWrapper(filtered_current_diagram)
    #             filtered_target_pi = TargetDiagramWrapper(filtered_target_diagram)

    #             # 6. Calculate the loss on the FILTERED diagrams
    #             class_loss = self.wasserstein_distance([filtered_current_pi], [filtered_target_pi])
    #             print(f"  6. Calculated Wasserstein Loss: {class_loss.item():.6f}")
                
    #             total_loss = total_loss + (class_loss.mean() * self.loss_scale_factor)
    #             valid_classes += 1
                
    #         except Exception as e:
    #             print(f"    !!! ERROR during loss calculation for {class_name}: {e}")
    #             continue
        
    #     if valid_classes > 0:
    #         final_loss = total_loss / valid_classes
    #         print(f"\n  >>> Final Batch Topological Loss: {final_loss.item():.6f}\n")
    #         return final_loss
    #     else:
    #         print("\n  >>> No valid classes, returning zero loss.\n")
    #         return torch.tensor(0.0, device=device, requires_grad=True)


    def forward(self, latent_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute topological loss based on the DIFFERENCE IN TOTAL PERSISTENCE.
        This provides a more stable gradient than Wasserstein distance.
        """
        total_loss = torch.tensor(0.0, device=latent_features.device, requires_grad=True)
        valid_classes = 0
        device = latent_features.device

        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (labels == class_idx)
            
            if class_mask.sum().item() < self.min_samples_per_class:
                continue
                
            class_features = latent_features[class_mask]
            
            try:
                # 1. Get the full PersistenceInformation object for the current batch.
                current_pi = self._compute_persistence_information(class_features)
                if current_pi is None or not hasattr(current_pi, 'diagram'):
                    continue

                # 2. Get the target diagram tensor.
                target_diagram_tensor = self.target_diagrams[class_name].diagram.to(device)

                # --- NEW: TOTAL PERSISTENCE LOSS CALCULATION ---

                # A. Calculate total persistence for the CURRENT diagram
                current_diagram_tensor = current_pi.diagram
                if current_diagram_tensor.shape[0] > 0:
                    current_total_persistence = torch.sum(current_diagram_tensor[:, 1] - current_diagram_tensor[:, 0])
                else:
                    current_total_persistence = torch.tensor(0.0, device=device)

                # B. Calculate total persistence for the TARGET diagram
                if target_diagram_tensor.shape[0] > 0:
                    target_total_persistence = torch.sum(target_diagram_tensor[:, 1] - target_diagram_tensor[:, 0])
                else:
                    target_total_persistence = torch.tensor(0.0, device=device)
                
                # C. The loss is the squared difference between the two totals.
                class_loss = (current_total_persistence - target_total_persistence) ** 2

                # --- END OF NEW LOGIC ---
                
                total_loss = total_loss + (class_loss * self.loss_scale_factor)
                valid_classes += 1
                
            except Exception as e:
                print(f"    WARNING: Loss calculation failed for {class_name}: {e}")
                continue
        
        if valid_classes > 0:
            return total_loss / valid_classes
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
