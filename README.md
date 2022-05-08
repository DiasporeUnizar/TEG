# TEG Library 
This repository includes the TEG library that has been created from the implementation of the TEG detectors.

## Structure of the repository
The core structure is the following:
- *dataset*: includes the dataset used in the library test and the usage examples
- *dist*: contains  ```.whl``` files, i.e., package saved in the ```Wheel``` format (the standard built-package format used for Python distributions). 
The last version of the TEG library can be directly installed   using ```pip install tegdet-2.0.0-py3-none-any.whl``` 
- *doc*: includes library documentation
- *examples*: includes scripts that use the API of the library and post-process the results
- *script_results*: includes all the result files from the test script and usage examples in ```csv``` format
- *tegdet*: includes the TEG detectors modules (see [TEG detectors](https://github.com/DiasporeUnizar/TEG/blob/master/doc/TEGdetectors.md))
- *test*: includes the test module (used to build and test the library)
- *.gitignore*
- *LICENSE*
- *README.md*
- *requirements.txt*: includes the ```Python``` required packages
- *setup.py*:  this file contains all the package metadata information. 
 
## How to install the library
The library can be used with ```Python_v3```.
It depends on the Python packages listed in the ```requirements.txt``` file which need to be installed using the command:

```$ pip install -r requirements.txt```

The library is local to this repository (not published yet in the official PyPI repository) and can be installed using the command:

```$ pip install dist/tegdet-2.0.0-py3-none-any.whl```


## User scripts
The ```example``` folder includes the following scripts:
 
- *tegdet_variants.py*
- *tegdet_params_sensitivity.py*: 
- *post_processing.py*

The first two scripts are examples of usage of the library APIs, they both rely on the dataset files in ```dataset``` folder, 
and produce a result file ```<name_of_the_script>_results.csv``` (with comma-separated values format) in the ```script_results``` folder.

The third one, relies on the files in the ```dataset``` folder and in the ```script_results``` folder to produce reports (comparison of the testing datasets, performance and  accuracy of the TEG-detectors).
Since the script generates 3D plots, it is necessary to install the ```matplotlib``` package before running it:

```$ pip install matplotlib```

The scripts can be run using the following command from the root directory of this repository:

```$ python3 examples/<name_of_the_script>.py```

## Test script
The ```test``` folder  includes the test script ```test_detector_comparer_TEG.py``` that is used during debugging to check the correctness of the library. 

## Reference




