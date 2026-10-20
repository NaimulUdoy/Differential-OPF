"""
Main execution script for Differentiable OPF Optimization Pipeline.
"""
import torch
from models.diff_opf import DifferentiableOPFLayer
from config import PD_NOMINAL

def main():
    print("=== Initializing Differentiable OPF Layer Pipeline ===")
    
    # Define optimization target variable (e.g., controllable load parameters to optimize)
    load_parameters = torch.tensor(PD_NOMINAL, dtype=torch.float32, requires_grad=True)
    
    # Instantiate the differentiable OPF layer
    opf_layer = DifferentiableOPFLayer()
    
    optimizer = torch.optim.Adam([load_parameters], lr=0.05)
    
    print("\nRunning gradient-based optimization over the OPF layer...")
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass through optimization layer
        p_gen_pred = opf_layer(load_parameters)
        
        # Define a dummy target loss function (e.g., penalize high generation dispatch)
        target_dispatch = torch.tensor([0.2, 0.8, 1.0], dtype=torch.float32)
        loss = torch.sum((p_gen_pred - target_dispatch) ** 2)
        
        # Backward pass utilizing implicit sensitivities / KKT gradients
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Updated Loads: {load_parameters.detach().numpy().round(3)}")

    print("\nOptimization completed successfully via differentiable programming!")

if __name__ == "__main__":
    main()