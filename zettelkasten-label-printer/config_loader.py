"""
Configuration Loader Module
Loads and manages application configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG = {
    'printer': {
        'model': 'QL-810W',
        'connection': 'usb://0x04f9:0x209b',
        'label_size': '62'
    },
    'fonts': {
        'h1_size': 48,
        'h2_size': 32,
        'body_size': 24,
        'page_size': 20,
        'font_family': 'Arial'
    },
    'layout': {
        'padding': 20,
        'line_spacing': 1.2,
        'max_label_height': 800
    },
    'data_sources': {
        'zotero_export_folder': '~/Documents/Zotero-Exports',
        'highlighted_export_folder': '~/Documents/Highlighted-Exports'
    },
    'options': {
        'auto_print': False,
        'save_preview': True,
        'preview_folder': '~/Documents/Label-Previews'
    }
}


class ConfigLoader:
    """Loads and manages application configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            print(f"Config file not found at {self.config_path}, using defaults")
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            # Merge with defaults (in case some keys are missing)
            config = DEFAULT_CONFIG.copy()
            if loaded_config:
                self._deep_update(config, loaded_config)

            return config

        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return DEFAULT_CONFIG.copy()

    def _deep_update(self, base: Dict, update: Dict) -> None:
        """
        Deep update base dictionary with values from update dictionary.

        Args:
            base: Base dictionary to update
            update: Dictionary with new values
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'printer.model')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: str = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save to (defaults to original config path)
        """
        save_path = Path(path) if path else self.config_path

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def expand_path(self, path: str) -> Path:
        """
        Expand user path (~/...) to absolute path.

        Args:
            path: Path string to expand

        Returns:
            Expanded Path object
        """
        return Path(path).expanduser().resolve()
