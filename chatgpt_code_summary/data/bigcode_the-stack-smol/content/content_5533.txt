from unittest import TestCase
from family_tree.member import Member
from family_tree import constants
from family_tree.family_tree import FamilyTree


class TestFamilyTree(TestCase):

    def setUp(self):
        self.ftree = FamilyTree()

    def test_add_child(self):
        result = self.ftree.add_child("Father", "Male")
        self.assertEqual(result, constants.CHILD_ADDITION_SUCCEEDED)
        self.assertEqual(
            self.ftree.family_tree.get("Father", None) is not None,
            True
        )

        self.assertEqual(
            self.ftree.add_child("Zim", "Male", "Mother"),
            constants.PERSON_NOT_FOUND
        )
        self.assertEqual(
            self.ftree.add_child("Zim", "Male", "Father"),
            constants.CHILD_ADDITION_FAILED
        )

        mother = Member(2, "Mother", "Female")
        mother.spouse = self.ftree.family_tree["Father"]
        self.ftree.family_tree["Father"].set_spouse(mother)
        self.ftree.family_tree["Mother"] = mother

        self.assertEqual(
            self.ftree.add_child("Zim", "Male", "Mother"),
            constants.CHILD_ADDITION_SUCCEEDED
        )
        self.assertEqual(
            self.ftree.add_child("Zim", "Male", "Mother"),
            constants.CHILD_ADDITION_FAILED
        )
        self.assertEqual(
            self.ftree.family_tree.get("Zim", None) is not None,
            True
        )

    def test_add_spouse(self):
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "Zim"),
            constants.SPOUSE_ADDITION_FAILED
        )
        dummy_member = Member(1, "DummyMember", "Male")
        self.ftree.family_tree['DummyMember'] = dummy_member
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "Zim"),
            constants.PERSON_NOT_FOUND
        )
        spouse_a = Member(1, "FakeMember", "Female")
        spouse_b = Member(1, "AlreadyMarriedMember", "Male")
        spouse_b.set_spouse(spouse_a)
        spouse_c = Member(1, "Zim", "Male")
        self.ftree.family_tree["FakeMember"] = spouse_a
        self.ftree.family_tree["AlreadyMarriedMember"] = spouse_b
        self.ftree.family_tree["Zim"] = spouse_c
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "FakeMember"),
            constants.SPOUSE_ADDITION_FAILED
        )
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "AlreadyMarriedMember"),
            constants.SPOUSE_ADDITION_FAILED
        )
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "Zim"),
            constants.SPOUSE_ADDITION_SUCCEEDED
        )
        self.assertEqual(
            self.ftree.add_spouse("Wife", "Female", "Zim"),
            constants.SPOUSE_ADDITION_FAILED
        )

    def test_get_relationship(self):
        self.assertEqual(
            self.ftree.get_relationship("Zim", "brother_in_law"),
            constants.PERSON_NOT_FOUND
        )
        member = Member(1, "Zim", "Male")
        son_a = Member(2, "SonA", "Male")
        son_b = Member(3, "SonB", "Male")
        member.add_child(son_b)
        member.add_child(son_a)
        son_a.set_father(member)
        son_b.set_father(member)
        self.ftree.family_tree["Zim"] = member
        self.ftree.family_tree["SonA"] = son_a
        self.ftree.family_tree["SonB"] = son_b
        self.assertEqual(
            self.ftree.get_relationship("Zim", "daughter"),
            constants.NONE
        )
        self.assertEqual(
            self.ftree.get_relationship("Zim", "son"), "SonA SonB")
