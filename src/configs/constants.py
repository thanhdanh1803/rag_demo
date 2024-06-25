import os


def _get_env(env_name: str, default=None, required=False):
    value = os.getenv(env_name)
    if value is None and required:
        raise ValueError(f"{env_name} env is required")
    return value or default


LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
USERNAME = _get_env("USERNAME", "admin", False)
PASSWORD = _get_env("PASSWORD", "admin", False)
API_KEY = _get_env("API_KEY", None, False)

AWS_REGION = _get_env("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY_ID = _get_env("AWS_ACCESS_KEY_ID", None, False)
AWS_SECRET_ACCESS_KEY = _get_env("AWS_SECRET_ACCESS_KEY", None, False)
AWS_OPENSEARCH_DOMAIN = "demo-search"
AWS_OPENSEARCH_URL = "search-demo-vector-euixqkbwguqw2asissoroz2fva.ap-southeast-1.es.amazonaws.com"

OPENAI_MODEL = "gpt-3.5-turbo"
