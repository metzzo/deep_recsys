"""
    {
        "name": "simple",
        "hidden_dim": 128,
        "patience": 100,
        "batch_size": 128,
        "num_epochs": 1, # 1000
        "config_constraint": {
            "max_dataset_size": 500000
        }
    },
    {
        "name": "complex",
        "hidden_dim": 128,
        "patience": 6,
        "batch_size": 64,
        "num_epochs": 25,
        "reduce_patience": 2,
        "num_gru_layers": 1,
        "fc_layer_size": 200,
        "phases": ['train', 'val']
    },
]

"""

recommender_configs = [
    {
        "name": "complex",
        "hidden_dim": 128,
        "patience": 100,
        "batch_size": 64,
        "num_epochs": 500,
        "reduce_patience": 20,
        "num_gru_layers": 1,
        "fc_layer_size": 200,
        "phases": ['train', 'val']
    },
]


def prepare_config(config):
    if config is None:
        config = {}

    return {
        'batch_size': config.get('batch_size') or 128,
        'patience': config.get('patience') or 100,
        'hidden_dim': config.get('hidden_dim') or 128,
        'num_epochs': config.get('num_epochs') or 500,
        'learning_rate': config.get('learning_rate') or 0.01,
        'phases': config.get('phases') or ['train', 'train_val', 'val'],
        'reduce_factor': config.get('reduce_factor') or 0.1,
        'reduce_patience': config.get('reduce_patience') or 75,
        'weight_decay': config.get('weight_decay') or 0.00001,
        'fc_layer_size': config.get('fc_layer_size') or 250,
        'num_gru_layers': config.get('num_gru_layers') or 1
    }

