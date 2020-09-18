# Useful Packages / Commands

I'll structure this section so that it exists as a resource more than anything; find the section that corresponds to the work (or task) that you are trying to do and find the packages that might be able to help. Please note that when we import modules into Python, all that we are really doing is importing new functions that we can use directly in our algorithms. There is nothing stopping us creating these functions ourselves and coding everything explicitly from the ground up in Python, just understand that that is an endeavour that will take a lot of time and these functions are available primarily at your convenience. The open-source support network in Python is one of the coding languages strongest attributes as users can share useful bits of code amongst each other with the goal of making everybody more productive. If you are coming from a MATLAB background, it may seem that MATLAB has all of the functionality built in. But that isn't completely the case, you may remember whenever you first install MATLAB that you have the option to select 'add-on'toolboxes that provide additional functionality such as the 'Control System Toolbox', Signal Processing Toolbox' etc, this is pretty much the same principle as what we are doing here.

## Preface: Importing Functions

Lets look at a simple example to see some of the options available to us when importing modules. We will use `numpy` which is the core library for scientific computing in Python. 

import numpy

Simply typing `import numpy` imports the entire `numpy` module into our Python environment. One of the useful functions associated with the `numpy` module is `linspace`. To access this function we type:

numpy.linspace(0,1,10)

Sometimes we might have the desire to change the package name that we import, perhaps instead of writing out `numpy` everytime we want to call a function from the `numpy` module; we might prefer to write `np` instead. This can be done easily by changing how we import the module using the `as` command:

import numpy as np

We can repeat the statement we performed previously in the same way but with the abbreviated name:

np.linspace(0,1,10)

You'll notice that it does the exact same thing but now we have a little more flexibility. When we import the numpy module we have access to all of the functions within it. You can imagine how this might introduce wastage into larger programs and could slow down runtimes, importing many functions that are unnecessary is not considered good practice when coding so here I show you an option to `import` specific functions directly. This is quite intuitive, we can say:

from numpy import linspace

So from the `numpy` module we want to import the `linspace` function. Importing the function in this direct way means that we do not have to call the function from the package. Instead, we have imported the `linspace` function directly and can then use it as such:

linspace(0,1,10)

```{Note}
You will see %%capture written in some of the code cells in this notebook. All this does is suppress the output of the code cell so that it does not print in the space below. It will just help to keep the notebook more condensed and easier to read.
```

~

## Data Manipulation

`numpy` `pandas`

~

`Numpy` is the core library for scientific computing in Python. It provides a high-performance multidimensional array object and tools for working with these arrays. Numpy is a powerful N-dimensional array object which is Linear algebra for Python. They allow us to work with vectors, matrices and arrays and manipulate them with ease. Full documentation at https://numpy.org/.

`Pandas` is an open-source library built on top of `numpy` providing high performance, easy-to-use data structures for data analysis. It allows for fast analysis, data cleaning and data preparation. It can work with data from a wide variety of sources and is suited for many kinds of data from a wide variety of sources; tabular data, time-series data and supports all data types.

~

### Numpy

#### Creating arrays

import numpy as np 
a = np.array([1,2,3])             # creates a 1D array
a = np.array([[1,2,3],[4,5,6]])   # creates a 2D array
a = np.arange(15)                 # generate 1D array with 15 elements from 0 to 14
a = np.arange(15).reshape(3,5)    # reshapes 1D array into 2D array with 3 rows and 5 columns

#### Understanding arrays

%%capture
a = np.array([1,2,3])             # creates a 1D array
print(a)                          # prints variable a
type(a)                           # returns the type of variable a
len(a)                            # returns the length of variable a
a.ndim                            # returns the number of dimensions of variable a
a.shape                           # returns the shape of variable a

#### Create more useful arrays

%%capture
np.linspace(0,3,4)                # creates 4 equally spaced points between 0 and 3
np.random.randint(0,100,6)        # create array of 6 random values between 0-100
np.random.rand(4)                 # create an array of uniform distribution (0,1)
np.eye(3)                         # create a 3*3 identity matrix
np.zeros(3)                       # create array [0,0,0]
np.zeros((5,5))                   # create 2D array with 5 rows and 5 columns of zeros
np.ones((2,3,4), dtype=np.int16)  # creates 3D array of size (2,3,4) with int16 datatype.

#### Basic Operations of ndarray

%%capture
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
A + B                             # addition of two arrays
A * B                             # element-wise product
A @ B                             # matrix product
A.T                               # transpose of matrix A
A.flatten()                       # form 1-D array
B < 3                             # creates Boolean matrix of B. True for elements < 3  
A.sum()                           # sum of all elements in A
A.sum(axis=0)                     # sum of each column
A.sum(axis=1)                     # sum of each row
A.min()                           # returns minimum value of A
A.max()                           # returns maximum value of A
np.exp(B)                         # returns exponential
np.sqrt(B)                        # returns square root of B
A.argmin()                        # position of minimum element in A

#### Indexing and Slicing 

%%capture
a = np.arange(4)**3               # creates 4 element vector a with values 0-3 cubed
a[2]                              # returns member of a in position 2
a[::-1]                           # reverses vector a
a[a>5]                            # returns a with values > 5

b = np.array([[2,0],[3,4], [3,1]])
b[0:2,1]                          # returns first two elements in column 1 (2nd colum)
b[:,0]                            # returns all elements in column 0 (1st column)

### Pandas









## Machine Learning

`keras` `tensorflow` `fastai` `pytorch`





## Operating System

`os`

~

The `os` module provides a portable way of using operating system dependent functionality. The documentation can be found here https://docs.python.org/3/library/os.html. The main function I use from this module is `os.getcwd()` which is useful for determining the current path you are working on in your current directory. This helps avoid any issues when you want to save logs, images or files into the correct places during your algorithm.

import os
path = os.getcwd()
print(path)

