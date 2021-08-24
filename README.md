# TEG Library 
This repository includes the TEG library that has been created from the implementation of the TEG detectors.

## Structure of the repository
The core structure is the following:
- *setup.py*:  this file  contains all the package metadata information. 
- *tegdet*: includes the TEG detectors modules  (see *TEG detectors implementation* below)
- *test*: includes the test module, the ```dataset``` folder (files related to one meterID, from ISSDA electricity dataset) and the ```script_results``` folder
- *dist*: contains a ```.whl``` file, i.e., package saved in the ```Wheel``` format (the standard built-package format used for Python distributions). 
The TEG library can be directly installed   using ```pip install tegdet-0.1.0-py3-none-any.whl``` 

 
## How to install the library
The library is local to this repository (not published yet) and can be installed using the command:

```$ pip install dist/tegdet-0.1.0-py3-none-any.whl```

## Usage
The ```test''' folder already includes an example of how to use the library, it is a revised version of the ```detector_comparer.py``` module.

## Reference
The creation of the library and the structure of the repository follows this  [guide](https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f).


## TEG detectors implementation
