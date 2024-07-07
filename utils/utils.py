import yaml
from models.CDCNs import CDCNpp

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg

def build_network(cfg, device='cuda: 0'):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None

    if cfg['model']['base'] == 'CDCNpp':
        network = CDCNpp(device=device)
    else:
        raise NotImplementedError

    return network
