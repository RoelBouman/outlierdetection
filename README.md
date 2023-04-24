##  Getting started with this benchmark

First, it pays to install all dependencies. The easiest way to do this is through the supplied .yml file through an Anaconda environment.
```
conda env create -f environment.yml
```

Then, activate the environment:
```
conda activate OD_benchmark
```

Due to permission (read/write) errors, it might be that the pip packages in the environment.yml file do not install correctly. It is then recommended to activate the OD_benchmark environment, and install these packages manually using `pip install` from within the environment.

If you want to run the DeepSVDD benchmark, or use the method in any other way, you also need to install a separate environment for DeepSVDD:

```
cd additional_methods/Deep-SVDD
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
cd ../..
```

The current installation instructions do not include GPU acceleration for the Tensorflow/PyTorch libraries. Should you wish to use it nonetheless, please follow the installation instructions for your specific system. Make sure to install these in the correct OD_benchmark conda environment.

When all dependencies are succesfully installed, the raw data needs to be downloaded:

### Include instructions on how to download data (included in current .zip file)

The raw data can then be processed using the `read_raw_write_in_format.py` script.

```
python3 read_raw_write_in_format.py
```

The pipeline can then be run using a nice CLI:

```
python3 preprocess_detect_outliers.py
```

or alternatively, you can add additional arguments to run only subsets. For example, you only want to run the kNN method on the wine dataset:

```
python3 preprocess_detect_outliers.py --method kNN --dataset wine
```

Reproducing the figures and analysis from the paper is then easily done using the following command:


```
python3 produce_figures.py
```


Extending the benchmark is easy! Datasets can be added by ensuring processed datafiles are added to the correct folders:
### include example of adding a dataset

Methods can easily be added, they only need to produce outlier scores and write them to files. If your implementation follows the Sklearn API, you can simply modify the following example:

### include example of adding a method
