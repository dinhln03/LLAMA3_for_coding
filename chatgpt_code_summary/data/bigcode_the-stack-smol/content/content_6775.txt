from flask import Flask
from flask import make_response
from flask import render_template
from flask import request
from flask import session

from blog_site.common.database import Database
from blog_site.webapp.models.blog import Blog
from blog_site.webapp.models.user import User

app = Flask(__name__)
app.secret_key = '\x1e\x14\xe6\xa0\xc5\xcc\xd9\x7f\xe5\xe8\x1cZ\xc5\xf2r\xb0W#\xed\xb6\xc8'


@app.route('/')
def home_temmplate():
    return render_template("home.html")


@app.route('/login')
def login_template():
    return render_template("login.html")


@app.route('/register')
def register_template():
    return render_template("register.html")


@app.before_first_request
def init_database():
    Database.initialize()


@app.route('/auth/login', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']

    if User.login_valid(email, password):
        User.login(email)
    else:
        session['email'] = None
        return render_template("login-error.html")

    return render_template("profile.html", email=session['email'])


@app.route('/auth/register', methods=['POST'])
def register_user():
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm-password']

    if password == confirm_password:
        User.register(email, password)
    else:
        # mismatch passwords
        # TODO: Insert validation error
        return render_template("register.html")

    return render_template("register-success.html", email=session['email'])


@app.route('/blogs/<string:user_id>')
@app.route('/blogs')
def user_blogs(user_id=None):
    blogs = None
    user = None
    if user_id is not None:
        user = User.get_by_id(user_id)
    else:
        if session['email'] is not None:
            user = User.get_by_email(session['email'])
            blogs = user.get_blogs()

    return render_template("user_blogs.html", blogs=blogs, email=user.email)


# TODO: User should be authenticated first before navigating to the post
@app.route('/posts/<string:blog_id>/')
def blog_posts(blog_id):
    blog = Blog.from_mongo_in_blog_object(blog_id)
    posts = blog.get_post()
    return render_template("user_blog_posts.html", blog_title=blog.title, blog_id=blog_id, posts=posts)


@app.route('/blogs/new/', methods=['GET', 'POST'])
def create_new_blog():
    if request.method == 'GET':
        return render_template("new_blog.html")
    else:
        title = request.form['title']
        description = request.form['description']
        user = User.get_by_email(session['email'])

        new_blog = Blog(user.email, title, description, user._id)
        new_blog.save_to_mongo()

        return make_response(blog_posts(user._id))


@app.route('/post/new/<string:blog_id>', methods=['GET', 'POST'])
def create_new_post(blog_id):
    if request.method == 'GET':
        return render_template("new_post.html", blog_id=blog_id)
    else:
        title = request.form['title']
        content = request.form['content']

        blog = Blog.from_mongo_in_blog_object(blog_id)
        blog.new_post(title, content)

        return make_response(blog_posts(blog_id))


if __name__ == '__main__':
    app.run(port=8660, debug='True')
