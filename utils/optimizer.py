import torch

__all__ = ["update_optimizer_param"]

def update_optimizer_param(
    optimizer: torch.optim.Optimizer,
    old_param: torch.nn.Parameter,
    new_param: torch.nn.Parameter,
) -> None:
    """
    Updates the optimizer to track a new parameter tensor instead of an old one,
    preserving the state if possible.

    Args:
        optimizer: The optimizer to update.
        old_param: The old parameter tensor (must be the exact object that was in the optimizer).
        new_param: The new parameter tensor.
    """
    found = False
    for group in optimizer.param_groups:
        params = group['params']
        for i, p in enumerate(params):
            if p is old_param:
                found = True
                params[i] = new_param

                # Transfer state
                if p in optimizer.state:
                    old_state = optimizer.state[p]
                    new_state = {}

                    for key, value in old_state.items():
                        # Handle tensor states (like exp_avg, exp_avg_sq) which match param shape
                        if torch.is_tensor(value) and value.shape == p.shape:
                            # We assume new_param is a superset of old_param (concatenated at end)
                            num_new = new_param.shape[0] - p.shape[0]
                            if num_new > 0:
                                # Create zeros with same properties as value (device, dtype)
                                # State tensors usually don't require grad
                                zeros = torch.zeros((num_new, *value.shape[1:]),
                                                  dtype=value.dtype,
                                                  device=value.device)
                                new_state[key] = torch.cat([value, zeros], dim=0)
                            else:
                                new_state[key] = value.clone()
                        else:
                            # For scalar tensors like 'step' or other types
                            new_state[key] = value.clone() if torch.is_tensor(value) else value

                    optimizer.state[new_param] = new_state
                    del optimizer.state[p]
                break
        if found:
            break
