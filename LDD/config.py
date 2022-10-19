dataloader_confg = {
    'CIFAR': {
        'train': {
            'batch_size': 250,
            'shuffle': True,
            'drop_last': False,
        },
        'test': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'valid': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
    },
    'CMNIST': {
        'train': {
            'batch_size': 250,
            'shuffle': True,
            'drop_last': False,
        },
        'test': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'valid': {
            'batch_size': 250,
            'shuffle': False,
            'drop_last': False,
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
    },
    'CelebA':{
        'train': {
            'batch_size': 125,
            'shuffle': True,
            'drop_last': False,
            'num_workers': 4,
            'prefetch_factor': 2,
        },
        'test': {
            'batch_size': 125,
            'num_workers': 4,
            'prefetch_factor': 2,
            'shuffle': False,
            'drop_last': False,
        },
        'valid': {
            'batch_size': 125,
            'num_workers': 4,
            'prefetch_factor': 2,
            'shuffle': False,
            'drop_last': False,
        },
        'memory': {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
            'num_workers': 4,
            'prefetch_factor': 2,
        }
    }
}