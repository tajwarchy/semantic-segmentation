import yaml
import os


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def print_config(config: dict):
    print("\n" + "="*45)
    print("         CONFIGURATION SUMMARY")
    print("="*45)
    for section, value in config.items():
        print(f"  {section:<22}: {value}")
    print("="*45 + "\n")