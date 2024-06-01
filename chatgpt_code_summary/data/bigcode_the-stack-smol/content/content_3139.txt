from django.contrib.auth import get_user_model

User = get_user_model()

superuser_username = 'admin'
superuser_email = 'superuser@admin.test'
superuser_password = 'admin_test'

try:
    User.objects.get(username=superuser_username)
except User.DoesNotExist:
    User.objects.create_superuser(
        superuser_username, superuser_email, superuser_password
    )
