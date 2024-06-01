# cash register

class RetailItem:

    def __init__(self, description, units_in_inventory, price):
        self.__description = description
        self.__units_in_inventory = units_in_inventory
        self.__price = price

    def get_description(self):
        return self.__description

    def get_units_in_inventory(self):
        return self.__units_in_inventory

    def get_price(self):
        return self.__price

    def set_description(self, description):
        self.__description = description

    def set_units_in_inventory(self, units_in_inventory):
        self.__units_in_inventory = units_in_inventory

    def set_price(self, price):
        self.__price = price

    def __str__(self):
        return self.__description + ", " + \
               "items: " + str(self.__units_in_inventory) + ", " + \
               "$ " + str(float(self.__price)) + ". "


class CashRegister:
    all_items_in_cart = []

    def purchase_item(self, retail_item):
        return self.all_items_in_cart.append(retail_item)

    def get_total(self):
        total = 0
        for item in self.all_items_in_cart:
            total += RetailItem.get_price(item) * \
                     RetailItem.get_units_in_inventory(item)
        return total

    def get_num_items(self):
        num_items = 0
        for item in self.all_items_in_cart:
            num_items += RetailItem.get_units_in_inventory(item)
        return num_items

    def show_items(self):
        if not self.all_items_in_cart:
            print("Your Cart is empty.")

        print("Your Cart:")
        for item in self.all_items_in_cart:
            print(item)

    def clear(self):
        self.all_items_in_cart.clear()


def main():
    more = "y"
    while more == "y":
        # if you want to manually enter object parameters manually
        # enter the instance attribute values
        print("If you want to manually enter object parameters\n"
              "manually enter the instance attribute values.")
        print("Enter yours item.")
        more_item = "y"
        while more_item == "y":
            description = input("description: ")
            units_in_inventory = int(input("count of item: "))
            price = float(input("price: "))
            items = RetailItem(description, units_in_inventory, price)
            CashRegister().purchase_item(items)

            more_item = input("More item yes -'y', no -'any ch'.")
            if more_item == "y":
                continue

            ready = input("Ready to pay? yes 'y', no 'any ch'")
            if ready == "y":
                print()
                # show all item in basket
                CashRegister().show_items()

                # Showing the customer the number of selected products
                print("Numbers of items:", CashRegister().get_num_items())

                # returns the total cost of all RetailItem objects
                # stored in the object's internal list 'all_items_in_cart'
                print("Total = $",
                      '{:.2f}'.format(CashRegister().get_total()))
                print()
                print("Enter 'y' if you pay, no 'any ch'.")
                print("Enter 'c' if you want clean cart.")
                pay = input("Enter: ")
                if pay == "y":
                    print("Paid.")
                    print("Product sent.")
                    # clears the internal list 'all_items_in_cart' of the
                    # CashRegister object. After payment, we clear the shopping
                    # cart, that is, the internal list 'all_items_in_cart'
                    print("Shopping cart empty.", CashRegister().clear())
                    break

                if pay == "c":
                    # clears the internal list 'all_items_in_cart' of the
                    # CashRegister object. After payment, we clear the shopping
                    # cart, that is, the internal list 'all_items_in_cart'
                    print("we clear the shopping cart",
                          CashRegister().clear())

            else:
                print("The product remained in cart.")

        more = input("Add more products? yes 'y', no 'any ch'")


main()
