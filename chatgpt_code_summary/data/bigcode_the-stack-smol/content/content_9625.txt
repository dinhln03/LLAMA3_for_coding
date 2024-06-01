from django.test import TestCase

###############################################
## test resources
###############################################
class TestPlayerSerializers(TestCase):
    fixtures = ['auth.json', 'team.json']
    
    def test_registration(self):
         
        client = APIClient()
        
        user = {'username': 'martin',
             'first_name': 'Martin',
             'last_name': 'Bright',
             'email':'martin@abc.com',
             'password':'pwd123',
             'confirm_password':'pwd123',
             'birth_year':1983}
        response = client.post('/' + API_PATH + 'register', user, format='json')
        assert response.status_code == 201