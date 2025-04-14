import torch

class GreedyLR:
    """Implementation of Zeroth Order GreedyLR scheduler compatible with DeepSpeed ZeroOptimizer
    
    This scheduler works well after using the learning rate finder to establish a good initial LR.
    It will dynamically adjust the learning rate based on training loss trends:
    - When loss is consistently improving, it increases the learning rate
    - When loss stops improving, it decreases the learning rate
    
    Args:
        optimizer: The DeepSpeed optimizer (ZeroOptimizer)
        factor (float): Factor by which the learning rate will be increased/decreased. Default: 0.1
        patience (int): Number of steps with no improvement after which learning rate will be changed. Default: 10
        threshold (float): Threshold for measuring the improvement. Default: 1e-4
        cooldown (int): Number of steps to wait before resuming normal operation. Default: 0
        warmup (int): Number of steps to linearly increase learning rate from 0. Default: 0
        min_lr (float): A scalar. A lower bound on the learning rate. Default: 0
        max_lr (float): A scalar. An upper bound on the learning rate. Default: 10.0
        smooth (bool): Whether to use smoothing for loss values. Default: True
        window (int): Window size for smoothing loss values when using window-based smoothing. Default: 50
        smoothing_factor (float): Beta factor for exponential moving average when using EMA smoothing. Default: None
        update_interval (int): Update frequency (consider changing LR every N steps). Default: 1
        reset (int): Number of steps after which to reset scheduler state. Default: 0
        verbose (bool): If True, prints a message to stdout for each update. Default: False
        debug (bool): If True, prints extended debugging information. Default: False
    """
    
    def __init__(self, optimizer, factor=0.1, patience=10, threshold=1e-10, 
                 cooldown=0, warmup=0, min_lr=0, max_lr=10.0, smooth=True, 
                 window=50, smoothing_factor=None, update_interval=1,
                 reset=0, verbose=False, debug=False):
        # Check if we're in a distributed environment
        self.distributed = torch.distributed.is_initialized()
        self.global_rank = torch.distributed.get_rank() if self.distributed else 0
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        self.is_main_process = self.global_rank == 0
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.warmup = warmup
        self.smooth = smooth
        self.window = window
        self.smoothing_factor = smoothing_factor
        self.update_interval = max(1, update_interval)  # Ensure at least 1
        self.reset = reset
        self.verbose = verbose
        self.debug = debug
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.best_loss = float('inf')
        self.loss_window = []
        self.ema_loss = None
        self.steps_since_last_update = 0
        
        if factor >= 1.0 or factor <= 0:
            raise ValueError('Factor should be between 0 and 1')
            
        self.min_lrs = min_lr
        self.max_lrs = max_lr
        
        # Store initial learning rate as base_lr
        param_groups = optimizer.param_groups
        self.base_lrs = [group['lr'] for group in param_groups]
        
        if self.debug:
            print(f"GreedyLR initialized with: factor={factor}, patience={patience}, "
                  f"update_interval={update_interval}, current_lr={self.base_lrs[0]}")
    
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
            'loss_window': self.loss_window,
            'ema_loss': self.ema_loss,
            'steps_since_last_update': self.steps_since_last_update,
            'update_interval': self.update_interval
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
        
        # Handle new fields with backward compatibility
        if 'ema_loss' in state_dict:
            self.ema_loss = state_dict['ema_loss']
        if 'steps_since_last_update' in state_dict:
            self.steps_since_last_update = state_dict['steps_since_last_update']
        if 'update_interval' in state_dict:
            self.update_interval = state_dict['update_interval']
    
    def get_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, metrics=None, epoch=None):
        """Execute a step of the scheduler with optional metrics."""
        # First verify we can access the optimizer and its param_groups
        if not hasattr(self.optimizer, 'param_groups') or len(self.optimizer.param_groups) == 0:
            if self.debug or self.verbose:
                print(f"Warning: Cannot access optimizer parameters in GreedyLR")
            return [self.base_lrs[0]]  # Return original LR
            
        param_groups = self.optimizer.param_groups
        current_lr = param_groups[0]['lr']
        
        if metrics is None:
            return current_lr
        
        # In distributed training, we need to aggregate losses from all processes
        if self.distributed:
            # Convert metrics to tensor for all_reduce
            metrics_tensor = torch.tensor([metrics], device="cuda" if torch.cuda.is_available() else "cpu")
            # Sum metrics across all processes
            torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            # Average the metrics
            metrics = metrics_tensor.item() / self.world_size
            
            if self.debug and self.is_main_process:
                print(f"GreedyLR: Aggregated loss from {self.world_size} processes: {metrics:.6f}")
            
        # Track the raw loss
        if self.debug and self.is_main_process:
            print(f"GreedyLR raw loss: {metrics:.6f}, current LR: {current_lr:.8f}")
            # Check if DeepSpeed's optimizer is properly connected
            if hasattr(self.optimizer, '_parameter_names'):
                print(f"Connected to DeepSpeed ZeroOptimizer with {len(self.optimizer.param_groups)} param groups")
            
        # Calculate smoothed loss - only on main process to ensure consistency
        if self.smooth:
            if self.smoothing_factor is not None:
                # Use exponential moving average (EMA)
                beta = self.smoothing_factor
                if self.ema_loss is None:
                    self.ema_loss = metrics
                else:
                    self.ema_loss = beta * self.ema_loss + (1 - beta) * metrics
                current_loss = self.ema_loss
                if self.debug and self.is_main_process:
                    print(f"GreedyLR EMA loss: {current_loss:.6f}")
            else:
                # Use window-based average
                self.loss_window.append(metrics)
                if len(self.loss_window) > self.window:
                    self.loss_window.pop(0)
                current_loss = sum(self.loss_window) / len(self.loss_window)
                if self.debug and self.is_main_process:
                    print(f"GreedyLR window average loss: {current_loss:.6f} (window size: {len(self.loss_window)})")
        else:
            current_loss = metrics
        
        # Increment step counter
        self.steps_since_last_update += 1
        
        # Only consider updating the learning rate at specified intervals
        # AND only have the main process make the decision to avoid conflicts
        if self.steps_since_last_update >= self.update_interval:
            # Only perform the decision logic on the main process
            should_update_lr = False
            new_lr = current_lr
            
            if not self.distributed or self.is_main_process:
                # Check if loss is better or worse
                # Using a relative threshold that scales with the loss
                rel_threshold = self.threshold * current_loss if self.threshold > 0 else 0
                if current_loss < self.best_loss - rel_threshold:
                    # Loss has improved
                    improvement = self.best_loss - current_loss
                    self.best_loss = current_loss
                    self.num_good_epochs += 1
                    # Don't reset bad epochs to 0, gradually reduce instead
                    self.num_bad_epochs = max(0, self.num_bad_epochs - 1)
                    if self.debug:
                        print(f"GreedyLR: Loss improved by {improvement:.6f}, good_steps={self.num_good_epochs}, bad_steps={self.num_bad_epochs}")
                else:
                    # Loss has not improved
                    # Don't reset good epochs to 0, gradually reduce instead
                    self.num_good_epochs = max(0, self.num_good_epochs - 1)
                    self.num_bad_epochs += 1
                    if self.debug:
                        print(f"GreedyLR: Loss didn't improve, good_steps={self.num_good_epochs}, bad_steps={self.num_bad_epochs}")
                    
                    # Add plateau detection - if loss has been within a small range for a while
                    if self.smooth and len(self.loss_window) >= 20:  # Need enough samples to detect plateau
                        recent_losses = self.loss_window[-20:]
                        loss_range = max(recent_losses) - min(recent_losses)
                        # If losses are within a small range (plateau) for 20 steps
                        if loss_range < (self.threshold * 5) and self.num_bad_epochs > self.patience // 2:
                            if self.debug:
                                print(f"GreedyLR: Detected plateau with loss range {loss_range:.6f}, forcing exploration")
                            # Force increase by setting good_epochs higher
                            self.num_good_epochs = self.patience + 1
                            self.num_bad_epochs = 0
                
                # Handle cooldown
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                    self.num_good_epochs = 0
                    if self.debug:
                        print(f"GreedyLR: In cooldown, {self.cooldown_counter} steps remaining")
                
                # Handle warmup
                if self.warmup_counter > 0:
                    self.warmup_counter -= 1
                    self.num_bad_epochs = 0
                    if self.debug:
                        print(f"GreedyLR: In warmup, {self.warmup_counter} steps remaining")
                
                # Adjust learning rate based on performance
                if self.num_good_epochs > self.patience:
                    # Increase learning rate
                    old_lr = current_lr
                    new_lr = min(current_lr / self.factor, self.max_lrs)
                    
                    if new_lr > old_lr:  # Only update if it's actually increasing
                        should_update_lr = True
                        if self.verbose or self.debug:
                            print(f'GreedyLR increasing learning rate from {old_lr:.6f} to {new_lr:.6f}')
                        
                        self.cooldown_counter = self.cooldown
                        self.num_good_epochs = 0
                        self.best_loss = float('inf')  # Reset best loss after LR change
                    
                elif self.num_bad_epochs > self.patience:
                    # Decrease learning rate
                    old_lr = current_lr
                    new_lr = max(current_lr * self.factor, self.min_lrs)
                    
                    if new_lr < old_lr:  # Only update if it's actually decreasing
                        should_update_lr = True
                        if self.verbose or self.debug:
                            print(f'GreedyLR reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')
                        
                        self.warmup_counter = self.warmup
                        self.num_bad_epochs = 0
                        self.best_loss = float('inf')  # Reset best loss after LR change
            
            # Synchronize decision across all processes in distributed setting
            if self.distributed:
                # Create tensors for broadcasting
                update_tensor = torch.tensor([1 if should_update_lr else 0], device="cuda" if torch.cuda.is_available() else "cpu")
                lr_tensor = torch.tensor([new_lr], device="cuda" if torch.cuda.is_available() else "cpu")
                
                # Broadcast decision and new LR from rank 0 to all processes
                torch.distributed.broadcast(update_tensor, 0)
                torch.distributed.broadcast(lr_tensor, 0)
                
                # Extract values from tensors
                should_update_lr = update_tensor.item() == 1
                new_lr = lr_tensor.item()
            
            # Apply the learning rate change on all processes if needed
            if should_update_lr:
                # Force update through the optimizer directly
                for param_group in param_groups:
                    param_group['lr'] = new_lr
                
                # Verify the change was applied on main process
                if not self.distributed or self.is_main_process:
                    actual_lr = param_groups[0]['lr']
                    if actual_lr != new_lr and (self.verbose or self.debug):
                        print(f"Warning: Failed to update LR! Expected {new_lr:.8f} but got {actual_lr:.8f}")
                        print("Attempting to update again with different method...")
                        # Try one more method - find the underlying optimizer
                        try:
                            # Access the internal optimizer if this is a DeepSpeed optimizer wrapper
                            if hasattr(self.optimizer, 'optimizer'):
                                print("Detected DeepSpeed optimizer wrapper, accessing internal optimizer")
                                internal_optimizer = self.optimizer.optimizer
                                for group in internal_optimizer.param_groups:
                                    group['lr'] = new_lr
                        except Exception as e:
                            print(f"Error updating internal optimizer: {e}")
            
            # Reset step counter after evaluation
            self.steps_since_last_update = 0
        
        # Check if we need to reset
        if self.reset > 0 and epoch is not None and epoch % self.reset == 0:
            if self.verbose or self.debug:
                print('GreedyLR resetting scheduler state')
            
            # Reset state
            self.best_loss = float('inf')
            self.num_good_epochs = 0
            self.num_bad_epochs = 0
            self.loss_window = []
            self.ema_loss = None
            
            # Reset learning rate to initial value
            for i, param_group in enumerate(param_groups):
                param_group['lr'] = self.base_lrs[i]
        
        return self.get_lr()
