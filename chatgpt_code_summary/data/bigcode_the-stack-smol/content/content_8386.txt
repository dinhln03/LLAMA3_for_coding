from project.customer import Customer
from project.equipment import Equipment
from project.exercise_plan import ExercisePlan
from project.subscription import Subscription
from project.trainer import Trainer


class Gym:
    def __init__(self):
        self.customers = []
        self.trainers = []
        self.equipment = []
        self.plans = []
        self.subscriptions = []

    def add_customer(self, customer: Customer):
        for objects in self.customers:
            if objects.name == customer.name:
                return
        self.customers.append(customer)

    def add_trainer(self, trainer: Trainer):
        for objects in self.trainers:
            if objects.name == trainer.name:
                return
        self.trainers.append(trainer)

    def add_equipment(self, equipment: Equipment):
        for objects in self.equipment:
            if objects.name == equipment.name:
                return
        self.equipment.append(equipment)

    def add_plan(self, plan: ExercisePlan):
        for objects in self.plans:
            if objects.id == plan.id:
                return
        self.plans.append(plan)

    def add_subscription(self, subscription: Subscription):
        for objects in self.subscriptions:
            if objects.id == subscription.id:
                return
        self.subscriptions.append(subscription)

    def subscription_info(self, subscription_id: int):
        result = []
        for s in self.subscriptions:
            if s.id == subscription_id:
                result.append(repr(s))
        for c in self.customers:
            if c.id == subscription_id:
                result.append(repr(c))
        for t in self.trainers:
            if t.id == subscription_id:
                result.append(repr(t))
        for e in self.equipment:
            if e.id == subscription_id:
                result.append(repr(e))
        for p in self.plans:
            if p.id == subscription_id:
                result.append(repr(p))
        return '\n'.join(result)
