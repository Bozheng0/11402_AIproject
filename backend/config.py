"""
集中管理環境變數。所有設定都從 .env 讀，不要寫死在程式裡。
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AI Service（組員的 BERT 服務位置）
    ai_service_url: str = "http://localhost:8001"
    ai_timeout_sec: float = 30.0  # BERT 第一次推論可能慢

    # OpenAI（用來生成「人話」說明）
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    enable_llm_narrative: bool = True  # 沒 key 時可關掉

    # 一般設定
    frontend_origin: str = "http://localhost:5173"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
