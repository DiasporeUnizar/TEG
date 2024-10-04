# ```tegdet``` Library 
A Time-Evolving-Graph (TEG) is a sequence of graphs generated from an univariate time series.

The ```tegdet``` library includes a set anomaly detectors which rely on TEGs both for the generation of the *prediction model*, 
from a *training dataset*, and for the prediction of ouliers, from  *testing datasets*.


## Structure of the repository
The core structure is the following:
- *dataset*: includes the datasets used in the library test and the usage examples
- *dist*: contains the library distributions (in ```Wheel``` format and zipped ```tar.gz```). 
- *doc*: includes library documentation
- *examples*: includes scripts that use the API of the library and post-process the results
- *script_results*: includes all the result files from the test script and usage examples in ```csv``` format
- *tegdet*: includes the TEG detectors modules (see [TEG detectors](https://github.com/DiasporeUnizar/TEG/blob/master/doc/TEGdetectors.md))
- *test*: includes the test module (used to build and test the library)
- *.gitignore*
- *LICENSE*
- *README.md*: this text 
- *README.txt*: short readme
- *requirements.txt*: includes the ```Python3``` required packages 
- *setup.py*:  this file contains all the package metadata information. 
 
## How to install the library
The library can be used with ```Python3``` (version >=3.6.1).

Since the library depends on the ```pandas```, ```numpy``` and ```scipy```  Python packages, these 
should be installed before using the library.

The library dependencies are also listed in the ```requirements.txt``` and  all the necessary packages can be installed using the command:

```$ pip3 install -r requirements.txt```

The library can be easily installed from the [PyPi](https://pypi.org/project/tegdet/) repository using the command:

```$ pip3 install tegdet``` 

In case you clone (or download) this repository, you can install the library using the command:

```$ pip3 install dist/tegdet-<current-version>-py3-none-any.whl```


## User scripts
The ```example``` folder includes different examples (one per sub-folder) with some scripts. In particular:
 
- *tegdet_variants.py*
- *tegdet_params_sensitivity.py*: 
- *post_processing.py*

The first two scripts are examples of usage of the library APIs, they both rely on the dataset files in ```dataset/<example>``` sub-folder, 
and produce a result file ```<name_of_the_script>_results.csv``` (with comma-separated values format) in the ```script_results/<example>``` sub-folder.

The third one, relies on the files in the ```dataset/<example>```  sub-folder and in the ```script_results/<example>``` sub-folder 
to produce reports (comparison of the testing datasets, performance and  accuracy of the TEG-detectors).
Since some of the scripts generate 3D plots, it is necessary to install the ```matplotlib``` package before running it:

```$ pip3 install matplotlib```

The scripts can be run using the following command from the root directory of this repository:

```$ python3 examples/<sub-folder>/<name_of_the_script>.py```

Before running the scripts, remember to set the ```PYTHONPATH``` environment variable to the root directory of the ```tegdet``` project:

```export PYTHONPATH="/my_tegdet_directory"```

## Test script
The ```test``` folder  includes the test script ```test_detector_comparer_TEG.py``` that is used during debugging to check the correctness of the library. 

## References
S. Bernardi, R. Javierre, J. Merseguer, *tegdet: An extensible Python library for anomaly detection using time evolving graphs*, SoftwareX, 101363 (2023), DOI: https://doi.org/10.1016/j.softx.2023.101363.

S. Bernardi, J. Merseguer, R. Javierre, *tegdet: An extensible Python Library for Anomaly Detection using Time-Evolving Graphs*, 
[CoRR abs/2210.08847 (2022)](https://arxiv.org/abs/2210.08847).


## Contributors 
- Simona Bernardi
- Raúl Javierre
- José Merseguer
- Ángel Villanueva
