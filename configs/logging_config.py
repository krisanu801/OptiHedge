import logging
import logging.config
import os
from typing import Dict, Any

def setup_logging(default_path: str = 'configs/logging.yaml', default_level: int = logging.INFO, env_key: str = 'LOG_CFG') -> None:
    """
    Setup logging configuration

    Args:
        default_path (str): Path to the logging configuration file.
        default_level (int): Default logging level if no configuration file is found.
        env_key (str): Environment variable key to check for a custom logging configuration file path.
    """
    path = os.getenv(env_key, default_path)
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path, 'r') as f:
                config: Dict[str, Any] = yaml.safe_load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            print(f"Error loading logging configuration from {path}: {e}. Using default level {default_level}.")
            logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Logging configuration file not found at {path}. Using default level {default_level}.")

if __name__ == '__main__':
    # Example Usage
    # 1. Create a logging.yaml file in the configs directory (see example below)
    # 2. Call setup_logging() at the beginning of your main script.
    # 3. You can override the default logging configuration by setting the LOG_CFG environment variable to the path of your custom logging configuration file.

    # Example logging.yaml file:
    # version: 1
    # formatters:
    #   simple:
    #     format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # handlers:
    #   console:
    #     class: logging.StreamHandler
    #     level: DEBUG
    #     formatter: simple
    #     stream: ext://sys.stdout
    #   file:
    #     class: logging.FileHandler
    #     level: INFO
    #     formatter: simple
    #     filename: logs/app.log
    #     mode: w
    # loggers:
    #   my_module:
    #     level: DEBUG
    #     handlers: [console, file]
    #     propagate: no
    # root:
    #   level: INFO
    #   handlers: [console, file]

    # Example usage in your main script:
    # from configs.logging_config import setup_logging
    # setup_logging()
    # logger = logging.getLogger(__name__)
    # logger.info("This is an info message.")
    # logger.debug("This is a debug message.")

    # Create a basic logger for this example
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully.")