from rolepermissions.roles import AbstractUserRole


class Admin(AbstractUserRole):
    available_permissions = {
        'deactivate_user': True,
        'activate_user': True,
        'change_user_permissions': True,
        'create_client_record': True,
        'delete_client_recods': True,
        'update_client_recods': True,
    }


class Manager(AbstractUserRole):
    available_permissions = {
        'create_client_record': True,
        'delete_client_recods': True,
        'update_client_recods': True,
    }
