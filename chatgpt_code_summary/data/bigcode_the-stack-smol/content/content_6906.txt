from flask import request
from flask_restplus import Resource

from app.project.auth import auth
from app.project.auth.auth_service import AuthService
from app.project.user.user_dto import UserDto
from app.project.user.user_service import UserService

api = UserDto.api
_user = UserDto.user


@api.route('/')
class UserList(Resource):
    @api.doc('list_of_registered_users')
    @api.marshal_list_with(_user, envelope='data')
    def get(self):
        """List all registered users"""
        return UserService.get_all_users()

    @auth.login_required
    @AuthService.admin_permission_required
    @api.response(201, 'User successfully created.')
    @api.doc('create a new user(only for admin)')
    @api.expect(_user, validate=True)
    def post(self):
        """Creates a new User(only for admin) """
        user_service = UserService()
        return user_service.create_user(request.json)


@api.route('/<public_id>')
@api.param('public_id', 'The User identifier')
@api.response(404, 'User not found.')
class User(Resource):
    @api.doc('get a user')
    @api.marshal_with(_user)
    def get(self, public_id):
        """get a user given its identifier"""
        user_service = UserService()
        user_service.load_user(public_id)
        if user_service.is_nan_user():
            api.abort(404)
        else:
            return user_service.get_user_public()
