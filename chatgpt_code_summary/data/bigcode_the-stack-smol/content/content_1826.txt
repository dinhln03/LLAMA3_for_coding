from flask import Flask, render_template, request, redirect
from flask import render_template

app = Flask(__name__)


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

from flask import Flask,request,render_template,redirect


# 绑定访问地址127.0.0.1:5000/user
@app.route("/user", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "user" and password == "password":
            return redirect("http://www.baidu.com")
        else:
            message = "Failed Login"
            return render_template('login.html', message=message)
    return render_template('login.html')


if __name__ == '__main__':
    app.run()



