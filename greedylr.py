import torch

class GreedyLR:
    """Implementation of Zeroth Order GreedyLR scheduler compatible with DeepSpeed ZeroOptimizer
    
    Args:
        optimizer: The DeepSpeed optimizer (ZeroOptimizer)
        factor (float): Factor by which the learning rate will be increased/decreased. Default: 0.1
        patience (int): Number of steps with no improvement after which learning rate will be changed. Default: 10
        threshold (float): Threshold for measuring the improvement. Default: 1e-4
        cooldown (int): Number of steps to wait before resuming normal operation. Default: 0
        warmup (int): Number of steps to linearly increase learning rate from 0. Default: 0
        min_lr (float): A scalar. A lower bound on the learning rate. Default: 0
        max_lr (float): A scalar. An upper bound on the learning rate. Default: 10.0
        smooth (bool): Whether to use exponential moving average for loss. Default: True
        window (int): Window size for smoothing loss values. Default: 50
        reset (int): Number of steps after which to reset scheduler. Default: 0
        verbose (bool): If True, prints a message to stdout for each update. Default: False
    """
    
    def __init__(self, optimizer, factor=0.1, patience=10, threshold=1e-4, 
                 cooldown=0, warmup=0, min_lr=0, max_lr=10.0, smooth=True, 
                 window=50, reset=0, verbose=False):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.warmup = warmup
        self.smooth = smooth
        self.window = window
        self.reset = reset
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.best_loss = float('inf')
        self.loss_window = []
        
        if factor >= 1.0 or factor <= 0:
            raise ValueError('Factor should be between 0 and 1')
            
        self.min_lrs = min_lr
        self.max_lrs = max_lr
        
        # Store initial learning rate as base_lr
        param_groups = optimizer.param_groups
        self.base_lrs = [group['lr'] for group in param_groups]
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'cooldown': self.cooldown,
            'warmup': self.warmup, 
            'best_loss': self.best_loss,
            'num_bad_epochs': self.num_bad_epochs,
            'num_good_epochs': self.num_good_epochs,
            'cooldown_counter': self.cooldown_counter,
            'warmup_counter': self.warmup_counter,
            'base_lrs': self.base_lrs,
            'loss_window': self.loss_window
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        self.factor = state_dict['factor']
        self.patience = state_dict['patience']
        self.threshold = state_dict['threshold']
        self.cooldown = state_dict['cooldown']
        self.warmup = state_dict['warmup']
        self.best_loss = state_dict['best_loss']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_good_epochs = state_dict['num_good_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.warmup_counter = state_dict['warmup_counter']
        self.base_lrs = state_dict['base_lrs']
        self.loss_window = state_dict['loss_window']
    
    def get_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, metrics=None, epoch=None):
        """Execute a step of the scheduler with optional metrics."""
        param_groups = self.optimizer.param_groups
        current_lr = param_groups[0]['lr']
        
        if metrics is None:
            return current_lr
            
        # Calculate smoothed loss if enabled
        if self.smooth:
            self.loss_window.append(metrics)
            if len(self.loss_window) > self.window:
                self.loss_window.pop(0)
            current_loss = sum(self.loss_window) / len(self.loss_window)
        else:
            current_loss = metrics
        
        # Check if loss is better or worse
        if current_loss < self.best_loss - self.threshold:
            # Loss has improved
            self.best_loss = current_loss
            self.num_good_epochs += 1
            self.num_bad_epochs = 0
        else:
            # Loss has not improved
            self.num_good_epochs = 0
            self.num_bad_epochs += 1
        
        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_good_epochs = 0
        
        # Handle warmup
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.num_bad_epochs = 0
        
        # Adjust learning rate based on performance
        if self.num_good_epochs > self.patience:
            # Increase learning rate
            new_lr = min(current_lr / self.factor, self.max_lrs)
            if self.verbose:
                print(f'GreedyLR increasing learning rate to {new_lr:.6f}')
            
            for param_group in param_groups:
                param_group['lr'] = new_lr
            
            self.cooldown_counter = self.cooldown
            self.num_good_epochs = 0
            
        elif self.num_bad_epochs > self.patience:
            # Decrease learning rate
            new_lr = max(current_lr * self.factor, self.min_lrs)
            if self.verbose:
                print(f'GreedyLR reducing learning rate to {new_lr:.6f}')
            
            for param_group in param_groups:
                param_group['lr'] = new_lr
            
            self.warmup_counter = self.warmup
            self.num_bad_epochs = 0
        
        # Check if we need to reset
        if self.reset > 0 and epoch is not None and epoch % self.reset == 0:
            if self.verbose:
                print('GreedyLR resetting scheduler state')
            
            # Reset state
            self.best_loss = float('inf')
            self.num_good_epochs = 0
            self.num_bad_epochs = 0
            self.loss_window = []
            
            # Reset learning rate to initial value
            for i, param_group in enumerate(param_groups):
                param_group['lr'] = self.base_lrs[i]
        
        return self.get_lr()
