from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str
    base_model: str = 'nvidia/nemotron-3-embed-1b:free'
    fallback_model: str = ''

    langsmith_api_key: str = ''
    langsmith_project: str = 'production-api'
    langsmith_endpoint: str = 'https://api.smith.langchain.com'
    langsmith_tracing_v2: bool = False


    app_env: str = 'development'
    log_level: str = 'INFO'
    rate_limit: int = 20 # 20 per minute
    cache_ttl_seconds: int = 3600
    max_retries: int = 3

    model_config: SettingsConfigDict = SettingsConfigDict(env_file=".env", extra="ignore")


    @property
    def is_production(self) -> bool:
        return self.app_env == 'production'


@lru_cache
def get_settings() -> Settings:
    return Settings()