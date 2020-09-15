# Glossary

Adapted from - https://www.codecademy.com/articles/glossary-python.

`github.com/samph4`

~ 

## Class

Python is a language that supports the Object Oriented Programming (OOP) paradigm. Like other OOP languages, python has classes which are defined wireframes of objects. Python supports class inheritance. A class may have many subclasses but may only inherit directly from one superclass.

### Syntax

class ClassName(object):
    """This is a class"""
    class_variable = 'some value'
    def __init__(self,*args):
        self.args=args
        
    def _repr_(self):
        return "Something to represent the object as a string"
    
    def other_method(self, *args):
        return "do something else"

### Example

class Horse(object):
    """Horse represents a horse"""
    species = "Equus ferus caballus"
    def _init_(self, color, weight, wild=False):
        self.color = color
        self.weight = weight 
        self.wild = wild
        
    def _repr_(self):
        return "%s horse weighing %f and wild status is %b" % (self.color, self.weight, self.wild)
    
    def make_sound(self):
        print("neighhhhh")
    
    def movement(self):
        return "walk"

## Comments

It's good practice to add comments to code to make it easier for others to understand what's happening. Code and algorithms can get messy, so it's good to augment code with human readable descriptions to explain design decisions. We do this simply by using the # key.

# we use the '#' key to make single line comments

Although sometimes we might wish to add a larger comment, that spans multiple lines. To do this, we use three apostrophes (speech marks also work).

"""
this is a much larger comment
that spans more than one line ~
It is useful when you want to comment
out multiple lines at once!
"""

```{tip}
You can also highlight multiple lines of code and use the command Ctrl-/ to quickly comment out the entire selection!
```

## print()

A useful function to display the output of a program.

print('some text here')

## range()

The `range()` function returns a list of integers, the sequence of which is defined by the arguments passed to it.

### Example

a = range(4)         #[0,1,2,3]
b = range(2,8)       #[2,3,4,5,6,7]
c = range(2,13,3)    #[2,5,8,11]

## Slice

Pythons way of indexing variables (like we do in MATLAB) is known as slicing and has both some similarities and some differences. One **key** difference is that indexing in Python begins at 0, and not 1. This means that if you wish to access the first element of an array, you refer to the 0th element rather than the 1st.

By 'slicing', we similarly use the square bracket that specifies the start and end of the section we wish to extract. Leaving the beginning value blank indicates you wish to start at the beginning, of the list, and leaving the ending value blank indicates that you wish to continue until the end of the list (this is the same for arrays and vectors). Using a negative value references the end of the list (so that in a list of 4 elements, -1 means the 4th element).

### Examples

x = [1,2,3,4]

#specifying a single element
x[1]

# specify beginning and end
x[2:4]

# specifying start at the next to last element and go to the end
x[-2:]

# specifying start at the beginning and go to the next to last element
x[:-1]

# specifying a step argument returns every nth item
y = [1,2,3,4,5,6,7,8]
y[::2]

# return a reversed version of the list (or string)
x[::-1]

# string reverse
my_name = 'samuel thompson'
my_name[::-1]

## Strings

Strings store characters and have many built-in convenience methods that let you modify their content. Strings are immutable, meaning they cannot be changed in place.

### Example

string1 = 'this is a valid string'
string2 = "this is also a valid string"   # note speech marks this time
string3 = 'this is' + ' ' + 'also' + ' ' + 'a string'
string3

You can also include variables into string outputs by using f strings. See below:

a = 80

string = f'The man has a mass of', a,'kg.'
string

b = 50
c = 46

string = 'Jane is %.2f years old and her husband Jim is %d years old' % (b,c)
string

```{note}
It is common to create strings using the % format. This is useful for a few reasons; one of which is that you can define the variables at the end of the string to make things a little clearer  - but also, it allows you to define the precision of the variables. ~%d is an integer value, %.2f has 2 decimal places etc.
```

## Tuples

A Python data type that holds an ordered collection of values, which can be of any type. Python tuples are 'immutable', meaning that once they have been created they can no longer be modified.

### Example

# tuples
x = (1,2,3,4)
y = ('banana', 'eggs', 'cereal')

my_list = [1,2,3,4]
my_tuple = tuple(my_list)
my_tuple

```{tip}
You can convert a list to a tuple by using the Python default function tuple(). Simply pass the list within the tuple function.
```

## Variables

Similar to MATLAB, variables are defined simply by using the = operator. This is not to be confused by the == operator which is used to test the equality of two variables. A variable can hold almost any type of variable such as lists, dictionaries and functions.

### Example

x = 12                          #integer
y = 'this is a string'          #string
z = [1,2,3,4]                   #list

```{note}
In MATLAB defining z = [1,2,3,4] would return a vector with four elements, by default in Python this creates a list. In order to create arrays, use the numpy package instead!
```

## Miscellaneous

### Attribute

A module is a Python object with arbitrarily named attributes that you can bind and reference. Simply, a module is a file consisting of Python code and can define functions, classes and variables.

### Module

A module is a Python object with arbitrarily named attributes that you can bind and reference. Simply, a module is a file consisting of Python code and can define functions, classes and variables.











