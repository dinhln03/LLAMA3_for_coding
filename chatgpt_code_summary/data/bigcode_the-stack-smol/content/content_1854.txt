#!flask/bin/python

from app import app
from config import DEBUG_MODE

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE)
