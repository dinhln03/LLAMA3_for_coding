from os import environ
from os.path import join, dirname

# Third party imports
from flask_compress import Compress


class Config:
    """
    Common configurations
    """

    # private variable used by Flask to secure/encrypt session cookies
    SECRET_KEY = environ['SECRET_KEY']

    # get the root url and concantenate it with the client secrets file
    OIDC_CLIENT_SECRETS = join(dirname(__file__), "client_secrets.json")

    # test out login and registration in development without using SSL
    OIDC_COOKIE_SECURE = False

    # URL to handle user login
    OIDC_CALLBACK_ROUTE = "/oidc/callback"

    # what user data to request on log in
    OIDC_SCOPES = ["openid", "email", "profile"]

    OIDC_ID_TOKEN_COOKIE_NAME = "oidc_token"

    TESTING = False
    DEBUG = False
    CSRF_ENABLED = True  # protect against CSRF attacks
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Gzip compression allows to reduce the size of the response by 70-90%
    # Flask-Compress compresses the application’s response with gzip
    COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/xml',
                          'application/json', 'application/javascript']
    COMPRESS_LEVEL = 6
    COMPRESS_MIN_SIZE = 500

    CACHE_TYPE = 'simple'  # compare with memcached, redis, filesystem, etc.

    # 'datastore' does not require any additional configuration.
    DATA_BACKEND = 'datastore'  # alternatively 'cloudsql' or 'mongodb'

    # Google Cloud Project ID
    PROJECT_ID = environ['PROJECT_ID']

    # CloudSQL & SQLAlchemy configuration
    CLOUDSQL_USER = environ['CLOUDSQL_USER']
    CLOUDSQL_PASSWORD = environ['CLOUDSQL_PASSWORD']
    CLOUDSQL_DATABASE = environ['CLOUDSQL_DATABASE']
    DB_HOST_IP = environ['DB_HOST_IP']
    DB_HOST_PORT = environ['DB_HOST_PORT']

    # The CloudSQL proxy is used locally to connect to the cloudsql instance.
    # To start the proxy, use:
    #
    #   $ cloud_sql_proxy -instances=your-connection-name=tcp:3306
    #
    # Port 3306 is the standard MySQL port. If you need to use a different port,
    # change the 3306 to a different port number.

    # Alternatively, use a local MySQL instance for testing.
    LOCAL_SQLALCHEMY_DATABASE_URI = (
        'postgresql://{}:{}@127.0.0.1:5432/{}').format(CLOUDSQL_USER, CLOUDSQL_PASSWORD, CLOUDSQL_DATABASE)

    # When running on App Engine, a unix socket is used to connect to the cloudsql instance.
    LIVE_SQLALCHEMY_DATABASE_URI = ('postgresql://{}:{}@{}:{}/{}').format(
        CLOUDSQL_USER, CLOUDSQL_PASSWORD, DB_HOST_IP, DB_HOST_PORT, CLOUDSQL_DATABASE)

    SQLALCHEMY_DATABASE_URI = LIVE_SQLALCHEMY_DATABASE_URI if environ.get(
        'FLASK_ENV') == 'production' else LOCAL_SQLALCHEMY_DATABASE_URI

    # Mongo configuration
    # If using mongolab, the connection URI is available from the mongolab control
    # panel. If self-hosting on compute engine, replace the values below.
    # MONGO_URI = 'mongodb://user:password@host:27017/database'

    # Google Cloud Storage and upload settings.
    # You can adjust the max content length and allow extensions settings to allow
    # larger or more varied file types if desired.
    CLOUD_STORAGE_BUCKET = environ['CLOUD_STORAGE_BUCKET']
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

    # OAuth2 configuration.
    # This can be generated from the Google Developers Console at
    # https://console.developers.google.com/project/_/apiui/credential.
    #
    #  * http://localhost:8080/oauth2callback
    #  * https://flask-okta-1.appspot.com/oauth2callback.
    #
    # If you receive a invalid redirect URI error review you settings to ensure
    # that the current URI is allowed.
    # GOOGLE_OAUTH2_CLIENT_ID = \
    #     'your-client-id'
    # GOOGLE_OAUTH2_CLIENT_SECRET = 'your-client-secret'


class ProductionConfig(Config):
    """
    Production configurations
    """

    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """
    Development configurations
    """

    DEBUG = True
    SQLALCHEMY_ECHO = True


class TestingConfig(Config):
    """
    Testing configurations
    """

    TESTING = True


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}


def configure_app(app):
    """Multiple app configurations"""

    # Configure Compressing
    Compress(app)
