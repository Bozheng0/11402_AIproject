from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 同學 ai_server 預設 :8000
    ai_service_url: str = "http://localhost:8000"
    ai_timeout_sec: float = 30.0

    # 我跑 :8080 避免衝突
    port: int = 8080
    frontend_origin: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
