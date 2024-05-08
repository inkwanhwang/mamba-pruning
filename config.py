import yaml
from box import Box

def load_config(conf_url):
    with open(conf_url, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config