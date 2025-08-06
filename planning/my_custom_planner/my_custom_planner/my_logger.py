import logging
import os
from datetime import datetime

class MyLogger:

    def __init__(self, log_directory='llm_interactions_logs', filename_prefix='llm_log_'):
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filepath = os.path.join(self.log_directory, f'{filename_prefix}{timestamp}.log')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_filepath)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.info(f'LLM interaction log started at: {self.log_filepath}')

    def log_request(self, prompt: str):
        self.logger.info(f'LLM Request: {prompt}')

    def log_response(self, response: str):
        self.logger.info(f'LLM Response: {response}')

    def log_error(self, error_message: str):
        self.logger.error(f'LLM Error: {error_message}')

    def log_warning(self, warning_message: str):
        self.logger.warning(f'LLM Warning: {warning_message}')