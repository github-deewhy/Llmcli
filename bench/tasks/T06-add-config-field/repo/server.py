from settings import Settings


def describe(settings: Settings) -> str:
    return f"{settings.host}:{settings.port}"
