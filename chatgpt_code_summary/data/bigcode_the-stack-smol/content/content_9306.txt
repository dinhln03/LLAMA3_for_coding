import unittest
from _2015 import d22_wizard


class TestWizard(unittest.TestCase):
    # damage, heal, acBonus, manaRecharge
    def test_evaluateSpellEffects_effects_empty(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual([0, 0, 0, 0], effects)

    def test_evaluateSpellEffects_activeEffects_empty(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)

    def test_evaluateSpellEffects_effects_MagicMissile(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {"Magic Missile": 1}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual([4, 0, 0, 0], effects)

    def test_evaluateSpellEffects_activeEffects_MagicMissile(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {"Magic Missile": 1}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)

    def test_evaluateSpellEffects_effects_Drain(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {"Drain": 1}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual([2, 2, 0, 0], effects)

    def test_evaluateSpellEffects_activeEffects_Drain(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {"Drain": 1}
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)

    def test_evaluateSpellEffects_effects_Shield(self):
        w = d22_wizard.Wizard(0, 0)
        duration = 6
        activeEffects = {"Shield": duration}
        for i in range(duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual([0, 0, 7, 0], effects)

    def test_evaluateSpellEffects_activeEffects_Shield(self):
        w = d22_wizard.Wizard(0, 0)
        duration = 6
        activeEffects = {"Shield": duration}
        for i in range(1, duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual({"Shield": duration - i}, activeEffects, i)
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)

    def test_evaluateSpellEffects_effects_Poison(self):
        w = d22_wizard.Wizard(0, 0)
        activeEffects = {"Poison": 6}
        duration = 6
        activeEffects = {"Poison": duration}
        for i in range(duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual([3, 0, 0, 0], effects)

    def test_evaluateSpellEffects_activeEffects_Poison(self):
        w = d22_wizard.Wizard(0, 0)
        duration = 6
        activeEffects = {"Poison": duration}
        for i in range(1, duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual({"Poison": duration - i}, activeEffects, i)
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)

    def test_evaluateSpellEffects_effects_Recharge(self):
        w = d22_wizard.Wizard(0, 0)
        duration = 5
        activeEffects = {"Recharge": duration}
        for i in range(duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual([0, 0, 0, 101], effects)

    def test_evaluateSpellEffects_activeEffects_Recharge(self):
        w = d22_wizard.Wizard(0, 0)
        duration = 5
        activeEffects = {"Recharge": duration}
        for i in range(1, duration):
            effects = w._evaluateSpellEffects(activeEffects)
            self.assertEqual({"Recharge": duration - i}, activeEffects, i)
        effects = w._evaluateSpellEffects(activeEffects)
        self.assertEqual({}, activeEffects)


if __name__ == '__main__':
    unittest.main()
