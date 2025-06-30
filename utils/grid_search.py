def get_nb_grid_search(adapter, cfg):
    if adapter is None or adapter in ['ia3']:
        return 1
    elif adapter in ['lora', 'dora']:
        return len(cfg[adapter].rank)
    elif adapter in ['vera', 'pvera']:
        return len(cfg[adapter].lr)
    elif adapter in ['adaptformer', 'bottleneck']:
        return len(cfg[adapter].reduction_ratios)