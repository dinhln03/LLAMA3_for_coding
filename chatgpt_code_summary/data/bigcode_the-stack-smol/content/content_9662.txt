"""
Ex 012 - make an algorithm that reads the price of a product and shows it with 5% discount

"""

print('Discover how much is a product with 5% off discount')
print('-' * 50)

pp = float(input('Enter the product price: '))
pd = pp - (pp / 100) * 5
print('-' * 50)
print(f"The product price was {pp:.2f}, on promotion of 5% will cost {pd:.2f}")

input('Enter to exit')
