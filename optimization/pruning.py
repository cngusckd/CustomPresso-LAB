import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class Pruner:
    def __init__(self, model):
        self.model = model

    def prune_global_unstructured(self, amount=0.2):
        """
        Prune 'amount' fraction of minimal weights globally across all Conv2d layers.
        """
        print(f"Pruning: Global Unstructured (Amount: {amount})")
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        print("Pruning applied (Masked).")

    def prune_structured(self, amount=0.2):
        """
        Prune channels (L1 structured) from Conv2d layers.
        Note: This is risky without handling connectivity (e.g. residual adds).
        Simple YOLO structures might break if C2f channels mismatch.
        This function applies the mask but changing tensor shapes requires 'removing' and handling dependencies.
        For Phase 3 'From-Scratch', we will apply the mask.
        """
        print(f"Pruning: Structured L1 (Amount: {amount})")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0) # dim=0 = output channels

    def remove_pruning(self):
        """
        Make pruning permanent (apply mask to weights and remove buffer).
        """
        print("Making pruning permanent...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight_orig'): # Check if pruned
                    prune.remove(module, 'weight')
