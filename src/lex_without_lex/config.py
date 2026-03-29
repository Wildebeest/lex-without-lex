from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "LWL_", "env_file": ".env"}

    feed_url: str = "https://lexfridman.com/feed/podcast/"
    data_dir: Path = Path("data")

    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "ktrGUw7rURIQyMrQZqCu"

    b2_key_id: str = ""
    b2_application_key: str = ""
    b2_bucket_name: str = ""

    base_url: str = "https://lex-without-lex.fly.dev"
