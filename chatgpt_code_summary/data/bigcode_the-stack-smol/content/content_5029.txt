from flask import Flask
app = Flask(__name__, static_url_path='', static_folder='static')
app.config['DEBUG'] = True

@app.route('/')
def root():
  # Note: this is probably handled by the app engine static file handler.
  return app.send_static_file('index.html')

@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404
