"""
Test Sermin config module
"""
from sermin.config.module import Registry, Namespace, Setting, settings
from sermin.config.utils import parse_args

from .utils import SafeTestCase


class SettingsTest(SafeTestCase):
    def setUp(self):
        self.old_settings = settings._namespaces
        settings._clear()

    def tearDown(self):
        settings.__dict__['_namespaces'] = self.old_settings

    def test_settings_exists(self):
        self.assertIsInstance(settings, Registry)

    def test_create_namespace(self):
        settings.test = 'Test settings'
        self.assertIsInstance(settings.test, Namespace)
        self.assertEqual(settings.test._label, 'Test settings')

    def test_create_setting(self):
        settings.test = 'Test settings'
        settings.test.setting = Setting('Test setting')
        self.assertIsInstance(settings.test._settings['setting'], Setting)
        self.assertEqual(
            settings.test._settings['setting'].label, 'Test setting',
        )
        self.assertEqual(settings.test.setting, None)

    def test_set_setting(self):
        settings.test = 'Test settings'
        settings.test.setting = Setting('Test setting')
        settings.test.setting = 'Testing'
        self.assertEqual(settings.test.setting, 'Testing')

    def test_cannot_redefine_namespace(self):
        settings.test = 'Test settings'
        with self.assertRaisesRegexp(
            ValueError, r'^Namespaces cannot be redefined$',
        ):
            settings.test = 'Second assignment'

    def test_cannot_redefine_setting(self):
        settings.test = 'Test settings'
        settings.test.setting = Setting('Test setting')
        with self.assertRaisesRegexp(
            ValueError, r'^Settings cannot be redefined$',
        ):
            settings.test.setting = Setting('Second assignment')

    def test_setting_evaluates_bool(self):
        settings.test = 'Test settings'
        settings.test.setting = Setting('Test')
        settings.test.setting = False
        self.assertTrue(type(settings.test.setting), bool)
        self.assertFalse(settings.test.setting)


class ParseArgsTest(SafeTestCase):
    def test_empty(self):
        unnamed, named = parse_args('')
        self.assertIsInstance(unnamed, list)
        self.assertIsInstance(named, dict)
        self.assertEqual(len(unnamed), 0)
        self.assertEqual(len(named), 0)
