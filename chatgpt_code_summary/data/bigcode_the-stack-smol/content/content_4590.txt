"""                     Global and local Scopes

            Scopes and Namespaces

When an object is assigned to a variable    # a = 10

    that variable points to some object

        and we say that the variable (name) is bound to that object

That object can be accessed using that name in various parts of our code

# ###   I can't reference that (a) just anywhere in my code!

That variable name and it's binding (name and object) only "exist" in specific parts of our code

    The porton of code where that name/binding is defined, is called the lexical scope of the variable

    These bindings are stored in namespaces

    (each scope has its own namespace)




            The global scope

The global scope is essentially the module scope

It spans a single file only

There is no concept of a truly global (across all the modules in our app) scope in Python

The only exception to this are some of the built=in globally available objects, such as:
    True    False   None    dict    print


The built-in global variables can be used anywhere inside our module

    including inside any function

Global scopes are nested inside the built-in scope

                                Built-in Scope
            Module 1                               name spaces
            Scope   name                        var1    0xA345E
                    space                       func1   0xFF34A

                    Module 2
                    Scope   name
                            space

If I reference a variable name inside a scope and Python does ot find it in that scope's namespace


        Examples

module1.py      Python does not find True or print in the current (module/global) scope
print(True)     So, it looks for them in the enclosing scope -> build-in
                Finds them there  -> True

module2.py      Python does not find a or print in the current (module/global) scope
print(a)        So

"""