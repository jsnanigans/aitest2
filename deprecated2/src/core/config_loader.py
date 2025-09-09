import toml

def load_config(config_path='config.toml'):
    """Loads the configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
            return config
    except Exception as e:
        raise
