import unittest

from cloudmitigator_semantic import version


class VersionTestCase(unittest.TestCase):

    def test_proper_version_parsing(self):
        proper_version = version.Version("v5.1.2")
        self.assertEqual(proper_version.major, 5)
        self.assertEqual(proper_version.minor, 1)
        self.assertEqual(proper_version.patch, 2)

    def test_bump_major(self):
        major_version = version.Version("v1.3.2")
        major_version.bump_major()
        self.assertEqual(major_version.major, 2)
        self.assertEqual(major_version.minor, 0)
        self.assertEqual(major_version.patch, 0)

    def test_bump_minor(self):
        minor_version = version.Version("v1.3.2")
        minor_version.bump_minor()
        self.assertEqual(minor_version.major, 1)
        self.assertEqual(minor_version.minor, 4)
        self.assertEqual(minor_version.patch, 0)

    def test_bump_patch(self):
        patch_version = version.Version("v1.3.2")
        patch_version.bump_patch()
        self.assertEqual(patch_version.major, 1)
        self.assertEqual(patch_version.minor, 3)
        self.assertEqual(patch_version.patch, 3)

    def test_pre_release(self):
        pre_version = version.Version("v1.3.2-alpha")
        self.assertEqual(pre_version.version, "v1.3.2-alpha")

    def test_meta_data(self):
        meta_data = version.Version("v1.3.2-alpha+361nh")
        self.assertEqual(meta_data.version, "v1.3.2-alpha+361nh")

    def test_version_changed_true(self):
        version_changed = version.Version("v1.1.0")
        version_changed.bump_minor()
        self.assertTrue(version_changed.version_changed)

    def test_version_changed_false(self):
        version_changed = version.Version("v1.1.0")
        self.assertFalse(version_changed.version_changed)

    def test_return_original_version(self):
        original_version = version.Version("v1.1.0")
        original_version.bump_patch()
        self.assertEqual(original_version.original_version, "v1.1.0")
        self.assertEqual(original_version.version, "v1.1.1")


