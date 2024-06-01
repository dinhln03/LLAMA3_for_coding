from flask import Flask, request, redirect, render_template, url_for, flash, jsonify
import gridfs, random, uuid, os
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from datetime import datetime

app = Flask(__name__,
            static_url_path = '',
            static_folder = 'static',
            template_folder = 'templates')

app.config['SECRET_KEY'] = 'big secrets'    
photos = UploadSet('photos', IMAGES)
app.config['UPLOAD_FOLDER'] = 'images_store'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# SQL form items
class PostItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=False, nullable=False)
    storeItem = db.Column(db.String(80), unique=False, nullable=False)
    avalability = db.Column(db.String(80), unique=False, nullable=False)
    location = db.Column(db.String(80), unique=False, nullable=False)
    #time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return (id, name, storeItem, avalability, location)

db.create_all()

def get_posts():
    query = [i.__dict__ for i in PostItem.query.all()]
    for item in query:
        del item['_sa_instance_state']
    return query


# Render webpages
@app.route("/")
def render_index():
    return render_template("index.html", posts = get_posts())

@app.route('/about')
def render_about():
    return render_template('about.html')

@app.route("/upload/", methods=['GET', 'POST'])
def render_upload():
    # Get form data
    if request.method == 'POST':

        # Check if the form is empty
        item = ""
        if '--------' == request.form.get('storeItem'):
            redirected = redirect(url_for('render_upload'))
            flash('Please select store item.')
            return redirected
        elif 'Other' == request.form.get('storeItem'):
           item = request.form.get('Other')
        else:
            item = request.form.get('storeItem')

        if None is request.form.get('radio'):
            redirected = redirect(url_for('render_upload'))
            flash('Please select an availability option.')
            return redirected

        if '' == request.form.get('Name'):
            redirected = redirect(url_for('render_upload'))
            flash('Please enter a name.')
            return redirected

        if '' == request.form.get('location'):
            redirected = redirect(url_for('render_upload'))
            flash('Please enter a location.')
            return redirected

        if '' == request.form.get('store'):
            redirected = redirect(url_for('render_upload'))
            flash('Please enter a store.')
            return redirected

        if 'photo' not in request.files:
            redirected = redirect(url_for('render_upload'))
            flash('Please upload a photo.')
            return redirected

        file = request.files['photo']
        if '' == file.filename:
            redirected = redirect(url_for('render_upload'))
            flash('No photo selected')
            return redirected

        locationStr = request.form.get('location') + '-' + request.form.get('store')
        
        # Save to database
        post = PostItem(name = request.form.get('Name'), storeItem = item, avalability = request.form.get('radio'), location = locationStr)
        db.session.add(post)
        db.session.commit()
    
        # Save the photo in the upload folder
        photo = request.files['photo']
        path = os.path.join(app.config['UPLOAD_FOLDER'], str(post.id))
        photo.save(path)

        # Print test
        print(str(post.id) + post.storeItem + post.avalability)

        return redirect(url_for('render_index'))

    return render_template('upload.html')

if __name__ == '__main__':
  app.run('0.0.0.0', 3000)
