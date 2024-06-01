import os


class DefaultConfig:
    """ Bot Configuration """

    HOST = "0.0.0.0"
    PORT = 3978

    CONNECTION_NAME = os.environ.get("CONNECTION_NAME", "echo-bot")
    APP_ID = os.environ.get("MICROSOFT_APP_ID", "")
    APP_PASSWORD = os.environ.get("MICROSOFT_APP_PASSWORD", "")

    LUIS_APP_ID = os.environ.get("LUIS_APP_ID", "")
    LUIS_API_KEY = os.environ.get("LUIS_API_KEY", "")
    # LUIS endpoint host name, ie "westus.api.cognitive.microsoft.com"
    LUIS_API_HOST_NAME = os.environ.get(
        "LUIS_API_HOST_NAME", "westeurope.api.cognitive.microsoft.com"
    )

    LUIS_IS_DISABLED = True if os.environ.get("LUIS_IS_DISABLED", "False") == "True" else False

    # cosmos storage
    COSMOS_DB_SERVICE_ENDPOINT = os.environ.get("COSMOS_DB_SERVICE_ENDPOINT", "")
    COSMOS_DB_KEY = os.environ.get("COSMOS_DB_KEY", "")
    COSMOS_DB_DATABASE_ID = os.environ.get("COSMOS_DB_DATABASE_ID", "")
    COSMOS_DB_CONTAINER_ID = os.environ.get("COSMOS_DB_CONTAINER_ID", "")
