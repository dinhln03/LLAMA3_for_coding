import os
import unittest
import abc

from funkyvalidate.examples.existing_directory import ExistingDirectory
from funkyvalidate.examples.existing_file import ExistingFile
from funkyvalidate import InterfaceType, meets


form_path = lambda *parts: os.path.abspath(os.path.join(*parts)) 

test_dir = form_path(__file__, '..')
example_dir = form_path(test_dir, 'test_example_dir')
nonexistant_dir = form_path(test_dir, 'nonexistant')
test_init_file = form_path(test_dir, '__init__.py')

# test_dir = os.path.abspath(os.path.join(__file__, '..', 'tests'))
# example_dir = os.path.abspath(os.path.join(test_dir, 'test_example_dir'))
# nonexistant_dir = os.path.abspath(os.path.join(test_dir, 'nonexistant'))
# test_init_file  = os.path.abspath(os.path.join(test))

# Rebase current directory
os.chdir(test_dir)

class ExamplesTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(test_init_file))

    def test_existingdirectory(self):
        self.assertTrue(isinstance(example_dir, ExistingDirectory))
        self.assertFalse(isinstance(nonexistant_dir, ExistingDirectory))
        self.assertFalse(isinstance(test_init_file, ExistingDirectory))

        # Test constructor
        self.assertTrue(ExistingDirectory(example_dir) == example_dir)
        self.assertRaises(TypeError, ExistingDirectory, 321.321)
        self.assertRaises(TypeError, ExistingDirectory, [example_dir])
        self.assertRaises(IOError, ExistingDirectory, nonexistant_dir)
        self.assertRaises(IOError, ExistingDirectory, test_init_file)

    def test_existingfile(self):
        """Test the value-type interface for existing files."""
        self.assertTrue(isinstance(test_init_file, ExistingFile))
        self.assertFalse(isinstance(example_dir, ExistingFile))

        # Test constructor
        self.assertTrue(ExistingFile(test_init_file) == test_init_file)
        self.assertRaises(TypeError, ExistingFile, 12)
        self.assertRaises(IOError, ExistingFile, 'wargarbl')
        self.assertRaises(IOError,  ExistingFile, nonexistant_dir)





class MyInterface(InterfaceType):
    @abc.abstractproperty
    def first_name(self):
        pass

class YesClass(object):
    def __init__(self):
        pass
    first_name = "foo"
yes = YesClass()

class AlsoClass(object):
    def __init__(self):
        self.first_name = "bar"
also = AlsoClass()

class NoClass(object):
    pass
no = NoClass()

class WeirdClass(object):
    def __init__(self):
        self.first_name = abc.abstractmethod(lambda self: NotImplemented)
    first_name = "bazinga"
weird = WeirdClass()

class FirstChild(MyInterface):
    def __init__(self):
        self.other_stuff = "boo"
# can't instantiate FirstChild

class SecondChild(FirstChild):
    first_name = "fixed"
second_child = SecondChild()


# class Weirder(MyInterface):
#     first_name = abc.abstractmethod(lambda self: NotImplemented)
#     def __init__(self):
#         self.first_name = abc.abstractmethod(lambda self: NotImplemented)

class CommutativeFirst(InterfaceType):
    first_name = abc.abstractmethod(lambda self: NotImplemented)
class CommutativeSecond(CommutativeFirst):
    def __init__(self):
        pass
    first_name = "booo"
commutative = CommutativeSecond()
class CommutativeFails(CommutativeFirst):
    """This cannot be instantiated, even though the instance
    overrides first_name. I believe this to be buggy behavior, however,
    it is shared by abc.ABCMeta. (IE its not my fault).
    """
    def __init__(self):
        self.first_name = "boo"


class InterfaceTests(unittest.TestCase):
    """These test __instancecheck__ and __subclasscheck__, which depend on the meets function.
    """
    def test_myinterface_itself(self):
        self.assertFalse(meets(MyInterface, MyInterface))
        self.assertFalse(issubclass(MyInterface, MyInterface))
        self.assertRaises(TypeError, MyInterface)

    def test_also_class(self):
        """
        AlsoClass does not meet the interface as a class, but does once instantiated.
        """
        self.assertFalse(meets(AlsoClass, MyInterface))
        self.assertTrue(meets(also, MyInterface))
        self.assertTrue(isinstance(also, MyInterface))
        self.assertFalse(issubclass(AlsoClass, MyInterface))

    def test_yes_class(self):
        """Meets interface"""
        self.assertTrue(meets(YesClass, MyInterface))
        self.assertTrue(meets(yes, MyInterface))
        self.assertTrue(isinstance(yes, MyInterface))
        self.assertTrue(issubclass(YesClass, MyInterface))

    def test_no_class(self):
        """Does not meet interface."""
        self.assertFalse(meets(NoClass, MyInterface))
        self.assertFalse(meets(no, MyInterface))
        self.assertFalse(isinstance(no, MyInterface))
        self.assertFalse(issubclass(NoClass, MyInterface))

    def test_weird_class(self):
        """Meets interface as class, but not as instance.
        This is strange - not something that would normally ever happen."""
        self.assertTrue(meets(WeirdClass, MyInterface))
        self.assertFalse(meets(weird, MyInterface))
        self.assertFalse(isinstance(weird, MyInterface))
        self.assertTrue(issubclass(WeirdClass, MyInterface))

    def test_first_child_class(self):
        """First child inherits MyInterface, but does not implement
        it at all - so it can't be implemented."""
        self.assertFalse(meets(FirstChild, MyInterface))
        self.assertFalse(issubclass(FirstChild, MyInterface))
        self.assertRaises(TypeError, FirstChild)

    def test_second_child_class(self):
        """Meets the interface inherited from its parent."""
        self.assertTrue(meets(SecondChild, MyInterface))
        self.assertTrue(meets(second_child, MyInterface))
        self.assertTrue(isinstance(second_child, MyInterface))
        self.assertTrue(issubclass(SecondChild, MyInterface))

    def test_commutative(self):
        """
        AlsoClass does not meet the interface as a class, but does once instantiated.
        """
        self.assertFalse(meets(CommutativeFirst, MyInterface))
        self.assertTrue(meets(CommutativeSecond, MyInterface))
        self.assertTrue(meets(commutative, MyInterface))
        self.assertTrue(isinstance(commutative, MyInterface))
        self.assertFalse(issubclass(CommutativeFirst, MyInterface))
        self.assertTrue(issubclass(CommutativeSecond, MyInterface))
        self.assertRaises(TypeError, CommutativeFails)


if __name__ == "__main__":
    unittest.main()
