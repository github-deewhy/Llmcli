from settings import Settings


def load_settings(data: dict) -> Settings:
    return Settings(
        host=str(data.get("host", "localhost")),
        port=int(data.get("port", 8080)),
    )
