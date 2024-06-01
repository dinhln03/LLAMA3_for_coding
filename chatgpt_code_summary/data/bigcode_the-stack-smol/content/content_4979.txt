#pip install flask-bcrypt
from flask_bcrypt import Bcrypt

# Create the Hasher
bcrypt = Bcrypt()

hashed_pass = bcrypt.generate_password_hash('somethingSuperSecret')
print(hashed_pass)

wrong_check = bcrypt.check_password_hash(hashed_pass, 'wrongpass')
print(wrong_check)

right_check = bcrypt.check_password_hash(hashed_pass, 'somethingSuperSecret')
print(right_check)
