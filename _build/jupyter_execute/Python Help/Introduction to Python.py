### Introduction to Python

Python is a relatively easy language to learn as it shares a lot of similarities with MATLAB. When we want to create neural networks and ai algorithms, its much easier to use Python as the programming language as you have access to many more tools and support than you would have with MATLAB. Whilst MATLAB is a nice integrated environment that does has excellent support, you are often dependent on the packages that they release and the overall support for machine learning tools is definitely limited when compared to the Python development environment. Python is nice because you have access to many open-source libraries that have pre-written functions that make your life a lot easier.

A basic example of this is something like the linspace function in MATLAB, upon loading MATLAB you'll be able to type into the command window ' a = linspace(0,10,10)' and it will store a variable a and print it out. Python works a little differently. Out of the box the linspace function is not present in the python libraries, but instead you can 'import' a library that gives you that same functionality. In the cell block below, we import a module named 'numpy'. This is a very popular package which contains a lot of the data manipulation tools you typically find and use in MATLAB. In python at the top of the document, it is good practice to import all of the libraries that you will use in the rest of your algorithm.

There are a bunch of different libraries people use to develop machine learning algorithms, from PyTorch, TensorFlow, Keras, FastAI etc etc, they all have positive attributes and negative attributes and quite often its a matter of preference which one you should use. I tend to use Keras to develop my models as it uses a simple syntax to write the code (functions are elegantly written and easy to understand), it shared the most comparisons with MATLAB so made sense in my head and gives you all of the flexibility you could ever really need .

#### Importing Modules and Interacting with them

import numpy

Here we have imported numpy, and therefore have access to all of the functions that exist within the numpy module - see the documentation for all the info https://numpy.org/doc/stable/reference/routines.math.html. But a useful function that is built into Python is one called 'dir'. This function can be used to show the entire directory of a module and list all of the functions that exist within it. So if we call dir(numpy):

dir(numpy)

Then a large list appears with all of the functions that exist within that module, and you can then select the function that suits your needs. If you are unsure as to what any of the functions do, then often you can call the 'help' function. Which again is built into Python. So if we consider the linspace example again, we can call help(numpy.linspace)

help(numpy.linspace)

This is obviously useful as it explains what the function itself does, and how it works. I used to use this all the time when I coded in MATLAB and it makes my life a lot easier in Python too. It also shows a bunch of examples at the end which is useful. So then in a similar fashion to MATLAB, we can define a linearly-spaced vector with 10 elements between 0 and 10 called 'a' in Python as follows:

a = numpy.linspace(0,10,10)

We can then 'print' the value of a into the command line by simply typing 'print(a)'. Note that when you define a variable python will not immediately print it into the command window and as such there is no need for the ';' to terminate the output like you do in MATLAB. Here, we can use the semi-colon to write multiple lines of code on the same line as:

a = numpy.linspace(0,10,10); b = numpy.linspace(0,20,10) 
print(f'a = ', a)
print(f'b = ', b)

# Note: That random f at the start of the print function, is known as a formatted string.
# It is useful because it allows you to print a string and a variable easily.

#### Types of Variables you will find in Python

You probably noticed that I did something weird in the print function when printing the variable. This was just so that the output looks nicer but raises an important point about variable types in Python. In Python, a variable can exist in any of five key formats:

* Numeric (integer, float, complex number)
* Dictionary
* Boolean (True/False)
* Set
* Sequence Type (strings, list, tuple).

You may or may not end up using a lot of these. But it is good to have some understanding of what kind of variable you are interacting with, especially if ever you run into problems with your code and need to figure out why. Often a good place to start is just to double check what 'type' your variable actually is. With that in mind, lets look at a couple examples:

c = 4
c1 = 4.3 
d = 'This is an example of a string! '
e = True
f = '4'

Here we have defined a bunch of different variables, by using the 'type' function we can confirm what variable c is by simply writing type(c). We see that it is an integer as you'd expect. This is important because you can therefore interact with this variable as you would expect with any number i.e. add, multiply, subtract and divide etc etc.

type(c)

Here I'll  use those formatted strings again just to show the data types of the other variables.

print(f'Variable c is a',type(c),'.')
print(f'Variable c is a',type(c1),'.')
print(f'Variable c is a',type(d),'.')
print(f'Variable c is a',type(e),'.')
print(f'Variable c is a',type(f),'.')

Note that variables c and f are both equal to 4. However it is important to understand that they are different variable types. See what happens when we add them to themselves.

c2 = c + c 
f2 = f + f 

print(f'c + c = ',c2)
print(f'f + f = ',f2)

You'll notice that for the string case it has literally added the variable next to itself. This would obviously be problematic if you were writing some code that was expecting the value to be 8. Python is very flexible and is extremely good at working with different types of data allowing you to code up all sorts of funky things without the same limitations that MATLAB gives you. But it's important to be aware of this kind of stuff. Similarly, this example indicates the same thing - however this time it is more obvious that it is a string and (depending on the program) this operation might be something that we would actually want to do:

print(d+d)

#### Loops and Conditional Statements

If you're using MATLAB chances are you're very familiar with creating loops to do a bunch of different things. I'll do my best to explain how loops and things work in Python because it is very similar but with a few differences. I'll start with a couple of simple example and maybe spend a moment looking at it to see what is going on - then ill explain them in a bit more detail. 

for i in numpy.linspace(0,5,6):
    print(i)

for i in range(6): # note range function uses integer values, linspace creates floats (float includes decimal).
    print(i)

fruits = ['apple','banana','cherry','grapefruit']
for x in fruits:
    print(x)

##### The Break statement
We can stop the loop early once a condition has been met. Note that conditions can be used with the Boolean data type, so if something is 'True' or 'False' - do something!

fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
  if x == "banana":
    break

Note that the print function is present before the break, if this was the other way around the loop would not print 'banana'! You'll probably have noticed that the conditional equal statement in Python is written as '==' with two equal signs. Here is a list of other operations you might find yourself wanting to use:

* == 

operators = {'Operator': '==', 'Expression': 'Equal to'}
operators

import pandas as pd 

df = pd.DataFrame(np.array([['Operator','Expression'],['==','Equal to']]))

df

help(pd.DataFrame)


relatively easy to learn the language and if you have basic programming knowledge in any language than studying python in deep for deep learning is not required and it is better if you learn syntax and concepts while implementing deep learning rather than studying python explicitly. This tutorial is enough to start and understand the codes that will be implemented in later tutorials. We will discuss the libraries and framework which we use to make the code in those tutorials only. This will be just basic python covering loops, arrays, lists, conditional statements, etc. So in case you know it you can skip.

