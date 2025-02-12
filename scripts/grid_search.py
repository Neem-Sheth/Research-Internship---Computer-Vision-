import torch
from itertools import product
from scripts.train import train_model
from scripts.model import DistancePredictor
from scripts.evaluate import evaluate_model
import copy

def grid_search_cv(train_dataset, val_dataset, device, param_grid):
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    for params in param_combinations:
        print(f"\nTrying parameters: {params}")
        
        # Create model with current parameters
        model = DistancePredictor(
            hidden_sizes=params['hidden_sizes'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        # Train model
        val_loss = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            num_epochs=params['num_epochs'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size']
        )
        
        results.append({
            'params': params,
            'val_loss': val_loss
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = copy.deepcopy(model)
            
            # Save the best model
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'params': best_params,
                'val_loss': best_val_loss
            }, 'best_model_grid_search.pth')
    
    return best_model, best_params, results
