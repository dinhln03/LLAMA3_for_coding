import turtle
'''fix'''
def draw_rhombus(some_turtle):
    for i in range(1,3):
        some_turtle.forward(20)
        some_turtle.right(315)
        some_turtle.forward(20)
        some_turtle.right(225)

def draw_ribbon(some_turtle):
    some_turtle.forward(100)
    some_turtle.right(150)
    some_turtle.forward(30)
    some_turtle.right(240)
    some_turtle.forward(30)
    some_turtle.right(150)
    some_turtle.forward(100)
    some_turtle.right(240)

def draw_flower(some_turtle):
    for i in range(1,11):
        draw_rhombus(some_turtle)
        some_turtle.right(36)
    some_turtle.right(336)
    some_turtle.forward(50)

def draw_wreath():
    window = turtle.Screen()
    window.bgcolor("black")

    brad = turtle.Turtle()
    brad.shape("turtle")
    brad.color("green")
    brad.speed(0)
    for i in range(1,16):
        draw_flower(brad)

    charlie = turtle.Turtle()
    charlie.shape("turtle")
    charlie.color("red")
    charlie.speed(2)

    charlie.right(60)

    for i in range(1,3):
        draw_ribbon(charlie)
    
    window.exitonclick()

draw_wreath()
