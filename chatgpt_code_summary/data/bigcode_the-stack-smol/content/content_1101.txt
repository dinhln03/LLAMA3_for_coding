from django.db import models
import string, random, datetime
from profiles.models import UserProfile, Location, Surcharges, User
from decimal import *
from menu.models import Product, Entree, Pizza, PizzaTopping, Side
from localflavor.us.models import PhoneNumberField, USStateField, USZipCodeField


#modify this to check against prior conf orders.

def make_conf(length=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(length))

    
class Order(models.Model):

    customer = models.ForeignKey(User, blank=True, null=True)
    created_date = models.DateTimeField(auto_now=False, auto_now_add=True)
    stamped = models.BooleanField(default=False)
    stamped_time = models.DateTimeField(auto_now=True, auto_now_add=False, blank=True, null=True)
    complete = models.BooleanField(default=False)
    delivery = models.BooleanField(default=False)
    delivery_available = models.BooleanField(default=False)
    location = models.ForeignKey(Location, blank=True, null=True)
    total = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    subtotal = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    taxes = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    first_name = models.CharField(max_length=120, blank=True, null=True)
    last_name = models.CharField(max_length=120, blank=True, null=True)
    street_address = models.CharField(max_length=120, blank=True, null=True)
    city = models.CharField(max_length=120, blank=True, null=True)
    state = USStateField(blank=True, null=True)
    zipcode = USZipCodeField(blank=True, null=True)
    phone = PhoneNumberField(blank=True, null=True)
    email = models.EmailField(max_length=120, blank=True, null=True)
    note = models.TextField(max_length=1000,blank=True, null=True)
    conf_number = models.CharField(max_length=20, blank=True, null=True)
    
    #delivery charge needs to be separate from lines
    
    def get_delivery_charge(self):
        return Location.objects.get(id=str(self.location)).get_delivery_charge()
    
    
    def compute_subtotal(self):
        lineitems = list(OrderLineItem.objects.filter(order=self.id))
        delivery_charge = Location.objects.get(id=str(self.location)).get_delivery_charge()
        lines = []
        for lineitem in lineitems:
            lines.append(lineitem.line_price)    
        if self.delivery == True:
            pre_sub = sum(lines)
            subtotal = sum(lines) + delivery_charge
            return subtotal
        else:
            return sum(lines)
            
    
    def compute_taxes(self):
        subtotal = self.compute_subtotal()
        loc = Surcharges.objects.get(location=self.location).location
        tax_rate = Decimal(str(loc.get_tax_rate()))
        return Decimal(round(subtotal * tax_rate, 2)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    
    def compute_total(self):
        return Decimal(round(self.compute_subtotal() + self.compute_taxes(), 2)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    
    class Meta:
        ordering = ['-stamped_time']
    
    
    def __unicode__(self):
        return str(str(self.created_date) + ' ' + str(self.id)) + str(self.customer)
    

        (PIZZA, 'PIZZA'),
        (SIDE, 'SIDE'),
        (SOUP,'SOUP'),
        (SALAD,'SALAD'),
        (BREADSTICKS,'BREADSTICKS'),
        (PASTA,'PASTA'),
        (WINGS,'WINGS'),
        (SANDWICH,'SANDWICH'),
        (BEVERAGE,'BEVERAGE'),


    
class OrderLineItem(models.Model):
    order = models.ForeignKey(Order)
    product = models.ForeignKey('menu.Product')
    size = models.CharField(max_length=7, blank=True, null=True)
    PIZZA = 'PIZZA'
    SIDE = 'SIDE'
    SOUP = 'SOUP'
    SALAD = 'SALAD'
    BREADSTICKS = 'BREADSTICKS'
    PASTA = 'PASTA'
    WINGS = 'WINGS'
    SANDWICH = 'SANDWICH'
    BEVERAGE = 'BEVERAGE'
    ITEM_TYPES = (
        (PIZZA, 'PIZZA'),
        (SIDE,'SIDE'),
        (SOUP,'SOUP'),
        (SALAD,'SALAD'),
        (BREADSTICKS,'BREADSTICKS'),
        (PASTA, 'PASTA'),
        (WINGS, 'WINGS'),
        (SANDWICH,'SANDWICH'),
        (BEVERAGE, 'BEVERAGE'),
    )
    product_type = models.CharField(max_length=50, choices=ITEM_TYPES, default=PIZZA)
    qty = models.PositiveIntegerField(default=1)
    line_price = models.DecimalField(max_digits=20, decimal_places=2, blank=True, null=True)
    toppings = models.ManyToManyField(PizzaTopping, blank=True, null=True, related_name='topping')
    

    
    def get_price(self):
        if self.product_type == 'PIZZA':
            pizza_price = Pizza.objects.get(product_id=self.product, size=self.size).get_price()
            pricing = []
            pricing.append(pizza_price)
            for topping in self.toppings.all():
                pricing.append(topping.price)
            return sum(pricing)
        elif self.product_type == 'ENTREE':
            return Entree.objects.get(product_id=self.product, size=self.size).get_price()
        elif self.product_type == 'SIDE':
            return Side.objects.get(product_id=self.product, size=self.size).price
    
    
    def __unicode__(self):
        return str(self.product)
