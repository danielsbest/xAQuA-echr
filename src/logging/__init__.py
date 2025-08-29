import os
import json
import logging.config

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'logging.json')
LOG_DIR = os.path.dirname(__file__)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

for handler in config.get('handlers', {}).values():
    if 'filename' in handler:
        log_file_path = os.path.join(LOG_DIR, handler['filename'])
        log_dir_for_file = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir_for_file):
            os.makedirs(log_dir_for_file)
        handler['filename'] = log_file_path

logging.config.dictConfig(config)

def get_logger(name):
    return logging.getLogger(name)