import json

def modify():
    path = 'code/tests/transformers/b0-mil-focal.ipynb'
    with open(path, 'r') as f:
        nb = json.load(f)

    # I'll modify the specific cells to integrate Optuna.
    # We will identify cells by their content.
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from torch import optim' in source:
                # Add optuna imports
                cell['source'].insert(-1, "import optuna\n")
                cell['source'].insert(-1, "from optuna.pruners import MedianPruner\n")
            
            if 'seed = 42' in source:
                # Add Optuna config
                cell['source'].insert(-1, "OPTUNA_SUBSET_FRAC = 0.3\n")
                cell['source'].insert(-1, "N_OPTUNA_EPOCHS = 5\n")
                cell['source'].insert(-1, "N_TRIALS = 20\n")
            
            # The structure of objective requires encapsulating the training loop.
            # It's probably easier to just overwrite the entire notebook with a new one adapted for Optuna,
            # or just systematically replace the cells.

if __name__ == '__main__':
    modify()
