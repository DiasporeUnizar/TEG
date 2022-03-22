# TEG Library 
This repository includes the TEG library that has been created from the implementation of the TEG detectors.

## Structure of the repository
The core structure is the following:
- *dataset*: include datasets used in the library test and the usage examples
- *dist*: contains a ```.whl``` file, i.e., package saved in the ```Wheel``` format (the standard built-package format used for Python distributions). 
The TEG library can be directly installed   using ```pip install tegdet-0.1.0-py3-none-any.whl``` 
- *tegdet*: includes the TEG detectors modules (see [TEG detectors](https://github.com/DiasporeUnizar/TEG/blob/master/doc/TEGdetectors.md))
- *test*: includes the test module and the ```script_results``` folder
- *.gitignore*
- *LICENSE*
- *README.md*

- *setup.py*:  this file  contains all the package metadata information. 
 
## How to install the library
The library is local to this repository (not published yet in the official PyPI repository) and can be installed using the command:

```$ pip install dist/tegdet-0.1.0-py3-none-any.whl```

## Usage
The ```test``` folder already includes an example of how to use the library, it is a revised version of the ```detector_comparer.py``` module.

## Reference
The creation of the library and the structure of the repository follows this  [guide](https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f).



