# define a function, which accepts 2 arguments
def cheese_and_crackers(cheese_count, boxes_of_crackers):
	# %d is for digit
	print "You have %d cheeses!" % cheese_count
	print "You have %d boxes of crackers!" % boxes_of_crackers
	print "Man that's enough for a party!"
	# go to a new line after the end
	print "Get a blanket.\n"


print "We can just give the function numbers directly:"
# call the function defined above
# by passing plain numbers, 
# also called numeric constants
# or numeric literals
cheese_and_crackers(20, 30)


print "OR, we can use variables from our script:"
# a variable definition
# doesn't need a'def' beforehand
amount_of_cheese = 10
amount_of_crackers = 50

# call (use, invoke, run) the function by passing the above variables
# or vars, for short
cheese_and_crackers(amount_of_cheese, amount_of_crackers)


print "We can even do math inside too:"
# python interpreter first calculates the math
# then passes the results as arguments
cheese_and_crackers(10 + 20, 5 + 6)


print "And we can combine the two, variables and math:"
# python substitutes the vars with their values, then does the math,
# and finally passes the calculated results to the function
# literals(consts), variables, math - all those called expressions
# calculating math and substituting var with their vals are called 'expression evaluation'
cheese_and_crackers(amount_of_cheese + 100, amount_of_crackers + 1000)

#################################################################
# another way to call a function is using result of calling another function
# which could be a built-in or custom
# also, don't forget about so-called "splats", when a function can accept any amount of args

def pass_any_two(*args):
	print "There are %d arguments" % len(args)
	print "First: %r" % args[0]
	print "Second: %r" % args[1]
	return "%r %r" % (args[0], args[1])

# 1: constants
pass_any_two(1, 2)

# 2: variables
first = "f"
second = "s"
pass_any_two(first, second)

# 3: math of consts
pass_any_two(4 + 6, 5 + 8)

# 4: math of vars
a = 5
b = 6
pass_any_two(a + 8, b * 2)

# 5: more than two args
pass_any_two(1, 2, 3, 4)

# 6: built-in function call results
txt = "what is my length?"
pass_any_two(len(txt), txt)

# 7: custom (same) function call results
pass_any_two(0, pass_any_two)

# 8: call by alias (just another name)
pass_any_2 = pass_any_two
pass_any_2("alias", "called")

# 9: call by invoking buil-in __call__ method
pass_any_two.__call__("__call__", "invoked")

# 10: call by passing a list, converted to multiple arguments
pass_any_two(*["list", "converted", 3, 4])






