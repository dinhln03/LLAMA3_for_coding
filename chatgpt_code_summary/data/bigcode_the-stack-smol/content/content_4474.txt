from django.test import TestCase, override_settings

from api.tests.fixtures import patch_requests
from .. import client


@patch_requests
@override_settings(
    VBO_URI_TEMPLATE="http://api/bag/verblijfsobject/{landelijk_id}/")
class ClientTest(TestCase):
    def setUp(self):
        self.client = client.Client()

    def test_get_vbo(self):
        vbo = self.client.get_verblijfsobject('0363010000998532')

        self.assertIsNotNone(vbo)
        self.assertEquals('0363010000998532', vbo.landelijk_id)

    def test_get_vbo_follow_pand_relatie(self):
        vbo = self.client.get_verblijfsobject('0363010000998532')
        panden = self.client.get_panden(vbo)

        self.assertIsNotNone(panden)
        self.assertEquals(1, len(panden))
        self.assertEquals("31", panden[0].pand_status)

    def test_get_vbo_follow_beperking_relatie(self):
        vbo = self.client.get_verblijfsobject('0363010000758545')
        beperkingen = self.client.get_beperkingen(vbo)

        self.assertIsNotNone(beperkingen)
        self.assertEquals(1, len(beperkingen))
        self.assertEquals("HS", beperkingen[0].beperking)
