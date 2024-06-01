"""
A simple   Line   class.
NOTE: This is NOT rosegraphics -- it is your OWN Line class.

Authors: David Mutchler, Vibha Alangar, Matt Boutell, Dave Fisher,
         Mark Hays, Amanda Stouder, Aaron Wilkin, their colleagues,
         and Jacob Jarski.
"""  # DONE: 1. PUT YOUR NAME IN THE ABOVE LINE.

import math
import m1t_test_Line as m1t


###############################################################################
# IMPORTANT:
#   Your instructor will help you get started on this exercise.
###############################################################################

# -----------------------------------------------------------------------------
# DONE: 2.  Right-click on the  src  folder and
#               Mark Directory as ... Sources Root,
#           if you have not already done so.
#
#     Then, with your instructor, READ THE INSTRUCTIONS in file
#         m0_INSTRUCTIONS.txt
#     asking questions as needed.  Once you understand the instructions,
#     mark this _TODO_ as DONE.
# -----------------------------------------------------------------------------

###############################################################################
# NOTE: For ALL of the methods that you implement, the method is allowed
# to have additional side effects as needed by it and/or other methods.
###############################################################################


def main():
    """
    Calls the   TEST   functions in this module, but ONLY if the method
    to be tested has at least a partial implementation.  That is,
    a  TEST   function will not be called until you begin work
    on the code that it is testing.
    """
    if m1t.is_implemented('__init__'):
        run_test_init()
    if m1t.is_implemented('clone'):
        run_test_clone()
    if m1t.is_implemented('reverse'):
        run_test_reverse()
    if m1t.is_implemented('slope'):
        run_test_slope()
    if m1t.is_implemented('length'):
        run_test_length()
    if m1t.is_implemented('get_number_of_clones'):
        run_test_get_number_of_clones()
    if m1t.is_implemented('line_plus'):
        run_test_line_plus()
    if m1t.is_implemented('line_minus'):
        run_test_line_minus()
    if m1t.is_implemented('midpoint'):
        run_test_midpoint()
    if m1t.is_implemented('is_parallel'):
        run_test_is_parallel()
    if m1t.is_implemented('reset'):
        run_test_reset()


###############################################################################
# Students:
#   Do NOT touch the following  Point  class - it has no TO DO.
#   Do NOT copy code from the methods in this Point class.
#
#   DO  ** READ **  this Point class,
#     asking questions about any of it that you do not understand.
#
#   DO  ** CALL **  methods in this Point class as needed
#     in implementing and testing the methods of the  ** Line **  class.
#
#   IMPORTANT, IMPORTANT, IMPORTANT:
#     *** In your  ** Line **  class methods, you should NEVER have code
#     *** that a  ** Point **  class method could do for you.
###############################################################################
# The   Point   class (and its methods) begins here.
###############################################################################


class Point(object):
    """ Represents a point in 2-dimensional space. """

    def __init__(self, x, y):
        """ Sets instance variables  x  and  y  to the given coordinates. """
        self.x = x
        self.y = y

    def __repr__(self):
        """
        Returns a string representation of this Point.
        For each coordinate (x and y), the representation:
          - Uses no decimal points if the number is close to an integer,
          - Else it uses 2 decimal places after the decimal point.
        Examples:
           Point(10, 3.14)
           Point(3.01, 2.99)
        """
        decimal_places = 2  # Use 2 places after the decimal point

        formats = []
        numbers = []
        for coordinate in (self.x, self.y):
            if abs(coordinate - round(coordinate)) < (10 ** -decimal_places):
                # Treat it as an integer:
                formats.append('{}')
                numbers.append(round(coordinate))
            else:
                # Treat it as a float to decimal_places decimal places:
                formats.append('{:.' + str(decimal_places) + 'f}')
                numbers.append(round(coordinate, decimal_places))

        format_string = 'Point(' + formats[0] + ', ' + formats[1] + ')'
        return format_string.format(numbers[0], numbers[1])

    def __eq__(self, p2):
        """
        Defines == for Points:  a == b   is equivalent to  a.__eq__(b).
        Treats two numbers as "equal" if they are within 6 decimal
        places of each other for both x and y coordinates.
        """
        return (round(self.x, 6) == round(p2.x, 6) and
                round(self.y, 6) == round(p2.y, 6))

    def clone(self):
        """  Returns a new Point at the same (x, y) as this Point. """
        return Point(self.x, self.y)

    def distance_from(self, p2):
        """ Returns the distance this Point is from the given Point. """
        dx_squared = (self.x - p2.x) ** 2
        dy_squared = (self.y - p2.y) ** 2

        return math.sqrt(dx_squared + dy_squared)

    def halfway_to(self, p2):
        """
        Given another Point object p2, returns a new Point
        that is half-way between this Point and the given Point (p2).
        """
        return Point((self.x + p2.x) / 2,
                     (self.y + p2.y) / 2)

    def plus(self, p2):
        """
        Returns a Point whose coordinates are those of this Point
        PLUS the given Point.  For example:
            p1 = Point(500, 20)
            p2 = Point(100, 13)
            p3 = p1.plus(p2)
            print(p3)
        would print:   Point(600, 33)
        """
        return Point(self.x + p2.x, self.y + p2.y)

    def minus(self, p2):
        """
        Returns a Point whose coordinates are those of this Point
        MINUS the given Point.  For example:
            p1 = Point(500, 20)
            p2 = Point(100, 13)
            p3 = p1.minus(p2)
            print(p3)
        would print:   Point(400, 7)
        """
        return Point(self.x - p2.x, self.y - p2.y)


###############################################################################
# The   Line   class (and its methods) begins here.
###############################################################################
class Line(object):
    """ Represents a line segment in 2-dimensional space. """

    def __init__(self, start, end):
        self.start = start.clone()
        self.originalstart = start.clone()
        self.end = end.clone()
        self.originalend = end.clone()
        self.timescloned = 0

        """
        What comes in:
          -- self
          -- a Point object named  start
          -- a Point object named  end
        where the two Points are to be the initial start and end points,
        respectively, of this Line.

        What goes out: Nothing (i.e., None).

        Side effects:  MUTATEs this Line by setting two instance
        variables named:
          -- start
          -- end
        to CLONES of the two Point arguments, respectively.
        Other methods must maintain those instance variables as needed
        so that they always indicate the CURRENT start and end points
        of this Line.

        Also, initializes other instance variables as needed
        by other Line methods.

        Example:  This   __init__  method runs when one constructs
        a Line.  So the 3rd of the following statements
        invokes the   __init__   method of this Line class:
            p1 = Point(30, 17)
            p2 = Point(50, 80)
            line = Line(p1, p2)        # Causes __init__ to run

            print(line.start)          # Should print Point(30, 17)
            print(line.end)            # Should print Point(50, 80)
            print(line.start == p1)    # Should print True
            print(line.start is p1)    # Should print False

        Type hints:
          :type start: Point
          :type end:   Point
        """
        # ---------------------------------------------------------------------
        # DONE: 3.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def __repr__(self):


        """
        What comes in:
          -- self
        What goes out: Returns a string representation of this Line,
        in the form:
           Line[(x1, y1), (x2, y2)]
        Side effects: None.
        Note:  print(BLAH)  causes BLAH's __repr__ to be called.
               BLAH's __repr__ returns a string,
               which the  print  function then prints.

        Example:  Since the  print  function calls __repr__ on the
        object to be printed:
            p1 = Point(30, 17)
            p2 = Point(50, 80)
            line = Line(p1, p2)  # Causes __init__ to run

            # The following statement causes __repr__ to run,
            # hence should print: Line[(30, 17), (50, 80)]
            print(line)

        Type hints:
          :rtype: str
        """
        # ---------------------------------------------------------------------
        # We have already implemented this  __repr__  function for you.
        # Do NOT modify it.
        # ---------------------------------------------------------------------
        start = repr(self.start).replace('Point', '')
        end = repr(self.end).replace('Point', '')
        return 'Line[{}, {}]'.format(start, end)

    def __eq__(self, line2):
        """
        What comes in:
          -- self
          -- a Line object
        What goes out: Returns  True  if:
             this Line's start point is equal to line2's start point AND
             this Line's end point is equal to line2's end point.
           Returns  False  otherwise.
        Side effects: None.
        Note:  a == b   is equivalent to  a.__eq__(b).

        Examples:
            p1 = Point(30, 17)
            p2 = Point(50, 80)

            line1 = Line(p1, p2)
            line2 = Line(p1, p2)
            line3 = Line(p2, p1)

            print(line1 == line1)   # Should print: True
            print(line1 == line2)   # Should print: True
            print(line1 == line3)   # Should print: False

            line1.start = Point(0, 0)
            print(line1 == line2)   # Should now print: False

        Type hints:
          :type  line2: Line
          :rtype: bool
        """
        # ---------------------------------------------------------------------
        # We have already implemented this  __eq__  function for you.
        # Do NOT modify it.
        # ---------------------------------------------------------------------
        return (self.start == line2.start) and (self.end == line2.end)

    def clone(self):
        self.timescloned = self.timescloned + 1
        clone = Line(self.start, self.end)
        return clone
        """
        What comes in:
          -- self
        What goes out: Returns a new Line whose START is a clone of
          this Line's START and whose END is a clone of this Line's END.
        Side effects: None.

        Example:
            p1 = Point(30, 17)
            p2 = Point(50, 80)
            line1 = Line(p1, p2)
            line2 = line1.clone()

            print(line1)  # Should print: Line[(30, 17), (50, 80)]
            print(line2)  # Should print: Line[(30, 17), (50, 80)]
            print(line1 == line2)              # Should print: True
            print(line1 is line2)              # Should print: False
            print(line1.start is line2.start)  # Should print: False
            print(line1.end is line2.end)      # Should print: False

            line1.start = Point(11, 12)
            print(line1)  # Should print: Line[(11, 12), (50, 80)]
            print(line2)  # Should print: Line[(30, 17), (50, 80)]
            print(line1 == line2)  # Should now print: False

        Type hints:
          :rtype: Line
        """
        # ---------------------------------------------------------------------
        # DONE: 4.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def reverse(self):
        reversestart = self.end
        reverseend = self.start
        self.start = reversestart
        self.end = reverseend
        """
        What comes in:
          -- self
        What goes out: Nothing (i.e., None).
        Side effects: MUTATES this Line so that its direction is reversed
        (that is, its start and end points are swapped).
        ** Must NOT mutate its start and end points -- just SWAP them. **

        Examples:
            p1 = Point(30, 17)
            p2 = Point(50, 80)
            line1 = Line(p1, p2)
            line2 = line1.clone()

            print(line1)  # Should print: Line[(30, 17), (50, 80)]

            line1.reverse()
            print(line1)  # Should print: Line[(50, 80), (30, 17)]
            print(line1 == line2)    # Should print: False

            line1.reverse()
            print(line1 == line2)    # Should now print: True
        """
        # ---------------------------------------------------------------------
        # DONE: 5.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def slope(self):
        slopex = (self.end.x-self.start.x)
        slopey = (self.end.y-self.start.y)
        if slopex == 0:
            return math.inf
        else:
             return slopey/slopex

        """
        What comes in:
          -- self
        What goes out: Returns the slope of this Line, or
           math.inf
        if the line is vertical (i.e., has "infinite" slope).
        Side effects: None.

        Examples:
            p1 = Point(30, 3)
            p2 = Point(50, 8)
            line1 = Line(p1, p2)

            # Since the slope is (8 - 3) / (50 - 30) , which is 0.25:
            print(line1.slope())    # Should print [approximately]: 0.25

            line2 = Line(Point(10, 10), Point(10, 5))
            print(line2.slope())    # Should print:  inf

            # math.inf is NOT the STRING 'inf', so:
            print(line2.slope() == 'inf')   # Should print False

        Type hints:
          :rtype: float
        """
        # ---------------------------------------------------------------------
        # DONE: 6.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def length(self):
        length = math.sqrt(((self.start.x- self.end.x) ** 2) + ((self.start.y - self.end.y) ** 2))
        return length
        """
        What comes in:
          -- self
        What goes out: Returns the length of this Line.
        Side effects: None.

        Example:
            p1 = Point(166, 10)
            p2 = Point(100, 10)
            line1 = Line(p1, p2)

            # Since the distance from p1 to p2 is 66:
            print(line1.length())  # Should print: 66.0

            p3 = Point(0, 0)
            p4 = Point(3, 4)
            line2 = Line(p3, p4)
            print(line2.length())  # Should print about 5.0

        Type hints:
          :rtype: float
        """
        # ---------------------------------------------------------------------
        # DONE: 7.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def get_number_of_clones(self):
        return self.timescloned
        """
        What comes in:
          -- self
        What goes out:
          -- Returns the number of times that this Line has been cloned
               (via the   clone   method).
        Side effects: None.

        Example:
            line1 = Line(Point(500, 20), Point(100, 8))
            line2 = line1.clone()
            line3 = line1.clone()
            line4 = line3.clone()
            line5 = line1.clone()
            print(line1.get_number_of_clones())
            print(line2.get_number_of_clones())
            print(line3.get_number_of_clones())
            print(line4.get_number_of_clones())
            print(line5.get_number_of_clones())
        would print:
            3    [since there are three   line1.clone()  statements]
            0    [since there are no      line2.clone()  statements]
            1    [since there is one      line3.clone()  statement]
            0    [since there are no      line4.clone()  statements]
            0    [since there are no      line5.clone()  statements]

        Type hints:
          :rtype: int:
        """
        # ---------------------------------------------------------------------
        # DONE: 8.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

    def line_plus(self, other_line):

        """
        What comes in:
          -- self
          -- another Line object
        What goes out:
          -- Returns a Line whose:
              -- start is the sum of this Line's start (a Point)
                   and the other_line's start (another Point).
              -- end is the sum of this Line's end (a Point)
                   and the other_line's end (another Point).
        Side effects: None.

        Example:
            line1 = Line(Point(500, 20), Point(100, 8))
            line2 = Line(Point(100, 13), Point(400, 8))
            line3 = line1.line_plus(line2)
            print(line3)
        would print:   Line[(600, 33), (500, 16)]

        Type hints:
          :type  other_line: Line
          :rtype: Line:
        """
        # ---------------------------------------------------------------------
        # DONE: 9.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

        start = Point(self.start.x + other_line.start.x, self.start.y + other_line.start.y)
        end = Point(self.end.x + other_line.end.x, self.end.y + other_line.end.y)
        line_plus = Line(start, end)
        return line_plus

    def line_minus(self, other_line):
        """
        What comes in:
          -- self
          -- another Line object
        What goes out:
          -- Returns a Line whose:
              -- start is this Line's start (a Point)
                   minus the other_line's start (another Point).
              -- end is this Line's end (a Point)
                   minus the other_line's end (another Point).
        Side effects: None.

        Example:
            line1 = Line(Point(500, 20), Point(100, 8))
            line2 = Line(Point(100, 13), Point(400, 8))
            line3 = line1.line_minus(line2)
            print(line3)
        would print:   Line[(400, 7), (-300, 0)]

        Type hints:
          :type  other_line: Line
          :rtype: Line:
        """
        # ---------------------------------------------------------------------
        # DONE: 10.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

        start = Point(self.start.x - other_line.start.x, self.start.y - other_line.start.y)
        end = Point(self.end.x - other_line.end.x, self.end.y - other_line.end.y)
        line_minus = Line(start, end)
        return line_minus
    def midpoint(self):
        """
        What comes in:
          -- self
        What goes out: returns a Point at the midpoint of this Line.
        Side effects: None.

        Example:
            p1 = Point(3, 10)
            p2 = Point(9, 20)
            line1 = Line(p1, p2)

            print(line1.midpoint())  # Should print: Point(6, 15)

        Type hints:
          :rtype: Point
        """
        # ---------------------------------------------------------------------
        # DONE: 11.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------

        midpoint = Point((self.end.x + self.start.x)/2, (self.end.y + self.start.y)/2)
        return midpoint
    def is_parallel(self, line2):
        """
        What comes in:
          -- self
          -- another Line object (line2)
        What goes out: Returns  True  if this Line is parallel to the
          given Line (line2).  Returns  False  otherwise.
            *** SEE THE IMPORTANT NOTE BELOW, re ROUNDING numbers.
        Side effects: None.

        Examples:
            line1 = Line(Point(15, 30), Point(17, 50))  # slope is 10.0
            line2 = Line(Point(10, 10), Point(15, 60))  # slope is 10.0
            line3 = Line(Point(10, 10), Point(80, 80))  # slope is  7.0
            line4 = Line(Point(10, 10), Point(10, 20))  # slope is inf

            print(line1.is_parallel(line2))   # Should print: True
            print(line2.is_parallel(line1))   # Should print: True
            print(line1.is_parallel(line3))   # Should print: False
            print(line1.is_parallel(line4))   # Should print: False
            print(line1.is_parallel(line1))   # Should print: True
            print(line4.is_parallel(line4))   # Should print: True

        Type hints:
          :type  line2: Line
          :rtype: bool
        """
        selfslopex = (self.end.x - self.start.x)
        line2slopex = (line2.end.x - line2.start.x)

        if line2slopex == 0:
            if line2slopex == selfslopex:
                return True
            else:
                return False
        if selfslopex == 0:
            return False
        selfslope =((self.end.y - self.start.y)/(self.end.x - self.start.x))
        line2slope = ((line2.end.y - line2.start.y)/ (line2.end.x - line2.start.x))
        if round(line2slope, 10) == round(selfslope, 10):
            return True
        else:
            return False



        # ---------------------------------------------------------------------
        # DONE: 12.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------
        #######################################################################
        #
        # IMPORTANT: When you test whether two FLOATING POINT numbers
        #   are "equal", you must deal with the imprecision
        #   of floating-point arithmetic.  For example, in REAL arithmetic,
        #         1 / (24 * math.pi - 20 * math.pi)
        #   and
        #         3 / (72 * math.pi - 60 * math.pi)
        #   are equal.  But in FLOATING point arithmetic, they are:
        #         0.07957747154594767
        #   and
        #         0.07957747154594765
        #   respectively (hence NOT equal).
        #   Try it out if you don't believe me!
        #
        #######################################################################
        # IMPORTANT BOTTOM-LINE:  When you want to test whether two
        # FLOATING POINT numbers  a  and  b  are the same, as in this method,
        #   DON'T use:               a == b
        #   INSTEAD use:  round(a, 12) == round(b, 12)
        ########################################################################
        #
        # The latter compares the numbers rounded to 12 decimal places.
        # In the context of this exercise, doing so is adequate to ignore
        # floating-point errors while distinguishing numbers that really
        # are different from each other.
        #######################################################################

    def reset(self):
        self.start = self.originalstart
        self.end = self.originalend
        """
        What comes in:
          -- self
        What goes out: Nothing (i.e., None).
        Side effects: MUTATES this Line so that its start and end points
          revert to what they were when this Line was constructed.

        Examples:
            p1 = Point(-3, -4)
            p2 = Point(3, 4)
            line1 = Line(p1, p2)
            line2 = Line(Point(0, 1), Point(10, 20))

              ... [various actions, including some like these:]
            line1.start = Point(100, 300)
            line2.end = Point(99, 4)
            line1.reverse()

            # Should print: Line[(x1, y1), (x2, y2)] where (x1, y1) and
            # (x2, y2) are the CURRENT coordinates of line1's endpoints.
            print(line1)
            print(line2)  # Similarly for line2

            line1.reset()
            line2.reset()
            print(line1)  # Should print: Line[(-3, -4), (3, 4)]
            print(line2)  # Should print: Line[(0, 1), (10, 20)]
        """
        # ---------------------------------------------------------------------
        # DONE: 13.
        #   a. READ the above specification, including the Example.
        #        ** ASK QUESTIONS AS NEEDED. **
        #        ** Be sure you understand it, ESPECIALLY the Example.
        #   b. Implement and test this method.
        #        The tests are already written (below).
        #        They include the Example in the above doc-string.
        # ---------------------------------------------------------------------


###############################################################################
# The TEST functions for the  Line  class begin here.
#
# We have already written the TEST functions.  They all take the form:
#   -- m1t.run_test_BLAH()  # This runs OUR tests.
#   -- One more test (or set of tests) that came directly from the Example
#        in the specification.
###############################################################################


def run_test_init():
    """ Tests the   __init__   method of the Line class. """
    m1t.run_test_init()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(30, 17)
    p2 = Point(50, 80)
    line = Line(p1, p2)  # Causes __init__ to run

    print(line.start)  # Should print Point(30, 17)
    print(line.end)  # Should print Point(50, 80)
    print(line.start == p1)  # Should print True
    print(line.start is p1)  # Should print False

    print('The above should print:')
    print('  Point(30, 17)')
    print('  Point(50, 80)')
    print('  True')
    print('  False')


def run_test_clone():
    """ Tests the   clone   method of the Line class. """
    m1t.run_test_clone()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(30, 17)
    p2 = Point(50, 80)
    line1 = Line(p1, p2)
    line2 = line1.clone()

    print(line1)  # Should print: Line[(30, 17), (50, 80)]
    print(line2)  # Should print: Line[(30, 17), (50, 80)]
    print(line1 == line2)  # Should print: True
    print(line1 is line2)  # Should print: False
    print(line1.start is line2.start)  # Should print: False
    print(line1.end is line2.end)  # Should print: False

    line1.start = Point(11, 12)
    print(line1)  # Should print: Line[(11, 12), (50, 80)]
    print(line2)  # Should print: Line[(30, 17), (50, 80)]
    print(line1 == line2)  # Should now print: False

    print('The above should print:')
    print('  Line[(30, 17), (50, 80)]')
    print('  Line[(30, 17), (50, 80)]')
    print('  True')
    print('  False')
    print('  False')
    print('  False')
    print('  Line[(11, 12), (50, 80)]')
    print('  Line[(30, 17), (50, 80)')
    print('  False')


def run_test_reverse():
    """ Tests the   reverse   method of the Line class. """
    m1t.run_test_reverse()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(30, 17)
    p2 = Point(50, 80)
    line1 = Line(p1, p2)
    line2 = line1.clone()

    print(line1)  # Should print: Line[(30, 17), (50, 80)]

    line1.reverse()
    print(line1)  # Should print: Line[(50, 80), (30, 17)]
    print(line1 == line2)  # Should print: False

    line1.reverse()
    print(line1 == line2)  # Should now print: True

    print('The above should print:')
    print('  Line[(30, 17), (50, 80)]')
    print('  Line[(50, 80), (30, 17)')
    print('  False')
    print('  True')


def run_test_slope():
    """ Tests the   slope   method of the Line class. """
    m1t.run_test_slope()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(30, 3)
    p2 = Point(50, 8)
    line1 = Line(p1, p2)

    # Since the slope is (8 - 3) / (50 - 30) , which is 0.25:
    print(line1.slope())  # Should print [approximately]: 0.25

    line2 = Line(Point(10, 10), Point(10, 5))
    print(line2.slope())  # Should print:  inf

    # math.inf is NOT the STRING 'inf', so:
    print(line2.slope() == 'inf')  # Should print False

    print('The above should print:')
    print('  0.25 (approximately)')
    print('  inf')
    print('  False')


def run_test_length():
    """ Tests the   length   method of the Line class. """
    m1t.run_test_length()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(166, 10)
    p2 = Point(100, 10)
    line1 = Line(p1, p2)

    # Since the distance from p1 to p2 is 66:
    print(line1.length())  # Should print: 66.0

    p3 = Point(0, 0)
    p4 = Point(3, 4)
    line2 = Line(p3, p4)
    print(line2.length())  # Should print about 5.0

    print('The above should print:')
    print('  66.0')
    print('  5.0 (approximately)')


def run_test_get_number_of_clones():
    """ Tests the   get_number_of_clones   method of the Line class. """
    m1t.run_test_get_number_of_clones()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    line1 = Line(Point(500, 20), Point(100, 8))
    line2 = line1.clone()
    line3 = line1.clone()
    line4 = line3.clone()
    line5 = line1.clone()
    print(line1.get_number_of_clones())
    print(line2.get_number_of_clones())
    print(line3.get_number_of_clones())
    print(line4.get_number_of_clones())
    print(line5.get_number_of_clones())
    print('The above should print 3, then 0, then 1, then 0, then 0.')


def run_test_line_plus():
    """ Tests the   line_plus   method of the Line class. """
    m1t.run_test_line_plus()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    line1 = Line(Point(500, 20), Point(100, 8))
    line2 = Line(Point(100, 13), Point(400, 8))
    line3 = line1.line_plus(line2)
    print(line3)
    print('The above should print:  Line[(600, 33), (500, 16)]')


def run_test_line_minus():
    """ Tests the   line_minus   method of the Line class. """
    m1t.run_test_line_minus()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    line1 = Line(Point(500, 20), Point(100, 8))
    line2 = Line(Point(100, 13), Point(400, 8))
    line3 = line1.line_minus(line2)
    print(line3)
    print('The above should print:  Line[(400, 7), (-300, 0)]')


def run_test_midpoint():
    """ Tests the   midpoint   method of the Line class. """
    m1t.run_test_midpoint()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(3, 10)
    p2 = Point(9, 20)
    line1 = Line(p1, p2)

    print(line1.midpoint())  # Should print: Point(6, 15)

    print('The above should print:  Point(6, 15)')


def run_test_is_parallel():
    """ Tests the   is_parallel   method of the Line class. """
    m1t.run_test_is_parallel()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    line1 = Line(Point(15, 30), Point(17, 50))  # slope is 10.0
    line2 = Line(Point(10, 10), Point(15, 60))  # slope is 10.0
    line3 = Line(Point(10, 10), Point(80, 80))  # slope is  7.0
    line4 = Line(Point(10, 10), Point(10, 20))  # slope is inf

    print(line1.is_parallel(line2))  # Should print: True
    print(line2.is_parallel(line1))  # Should print: True
    print(line1.is_parallel(line3))  # Should print: False
    print(line1.is_parallel(line4))  # Should print: False
    print(line1.is_parallel(line1))  # Should print: True
    print(line4.is_parallel(line4))  # Should print: True

    print('The above should print:')
    print('  True,  True,  False,  False,  True,  True')


def run_test_reset():
    """ Tests the   reset   method of the Line class. """
    m1t.run_test_reset()  # This runs OUR tests.
    # -------------------------------------------------------------------------
    # One ADDITIONAL test (or set of tests).
    # -------------------------------------------------------------------------
    p1 = Point(-3, -4)
    p2 = Point(3, 4)
    line1 = Line(p1, p2)
    line2 = Line(Point(0, 1), Point(10, 20))

    line1.start = Point(100, 300)
    line2.end = Point(99, 4)
    line1.reverse()

    # Should print: Line[(x1, y1), (x2, y2)] where (x1, y1) and
    # (x2, y2) are the CURRENT coordinates of line1's endpoints.
    print(line1)
    print(line2)  # Similarly for line2

    line1.reset()
    line2.reset()
    print(line1)  # Should print: Line[(-3, -4), (3, 4)]
    print(line2)  # Should print: Line[(0, 1), (10, 20)]

    print('The above should print:')
    print('  Line[(3, 4), (100, 300)]')
    print('  Line[(0, 1), (99, 4)]')
    print('  Line[(-3, -4), (3, 4)]')
    print('  Line[(0, 1), (10, 20)]')


# -----------------------------------------------------------------------------
# If this module is running at the top level (as opposed to being
# imported by another module), then call the 'main' function.
# It is necessary here to enable the automatic testing in m1t_test_Line.py.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
