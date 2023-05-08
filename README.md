# Unsupervised anomaly detection algorithms on real-world data: how many do we need?
This is the repository supplementing our paper.
Currently, this is the largest benchmark of unsupervised anomaly detection algorithms, with 32 algorithms applied on 52 datasets. 

You can cite our paper as follows:

```
@article{bouman2023unsupervised,
  title={Unsupervised anomaly detection algorithms on real-world data: how many do we need?},
  author={Bouman, Roel and Bukhsh, Zaharah and Heskes, Tom},
  journal={arXiv preprint arXiv:2305.00735},
  year={2023}
}

```

## Running the full benchmark
In order to run the full benchmark, you will need to install all dependencies. The easiest way to do this is through the supplied .yml file through an Anaconda environment.
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

You can replace the Conda environment name `myenv` with any of your choice, but you will have to change the name accordingly in the `run_all_methods.py` script.

The current installation instructions do not include GPU acceleration for the Tensorflow/PyTorch libraries. Should you wish to use it nonetheless, please follow the installation instructions for your specific system. Make sure to install these in the correct OD_benchmark conda environment.

When all dependencies are succesfully installed, you can either re-run the preprocessing, or make use of the existing preprocessed `.npz` files.

If you want to download all raw data, you can download them from the following sources:

| **Name**    | **Source URL**                                                                                                                                                                                                                               | **Datasets**                                                                                                                                                                                                                                                                                                      |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ADBench     | https://github.com/Minqi824/ADBench/tree/main/datasets                                                                                                                                                                                       | 11_donors.npz, 12_fault.npz, 19_landsat.npz, 22_magic.gamma.npz, 33_skin.npz, 42_WBC.npz, 46_WPBC.npz, 47_yeast.npz, 4_breastw.npz, 5_campaign.npz                                                                                                                                                                |
| ELKI        | https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/                                                                                                                                                                                 | Hepatitis_withoutdupl_norm_16.arff, InternetAds_withoutdupl_norm_19.arff, PageBlocks_withoutdupl_norm_09.arff, Parkinson_withoutdupl_norm_75.arff, Stamps_withoutdupl_norm_09.arff, Wilt_withoutdupl_norm_05.arff                                                                                                 |
| extended AE | https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill, https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization, https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification | HRSS_anomalous_optimized.csv, HRSS_anomalous_standard.csv, nasa.csv, and the entire folder: "CNC-kaggle"                                                                                                                                                                                                          |
| GAAL        | https://github.com/leibinghe/GAAL-based-outlier-detection/blob/master/Data/                                                                                                                                                                  | Spambase, Waveform                                                                                                                                                                                                                                                                                                |
| Goldstein   | http://dx.doi.org/10.7910/DVN/OPQMVF                                                                                                                                                                                                         | aloi-unsupervised-ad.csv, pen-global-unsupervised-ad.csv, pen-local-unsupervised-ad.csv                                                                                                                                                                                                                           |
| ODDS        | http://odds.cs.stonybrook.edu/                                                                                                                                                                                                               | annthyroid.mat, arrhythmia.mat, cardio.mat, cover.mat, glass.mat, http.mat, ionosphere.mat, letter.mat, mammography.mat, mnist.mat, musk.mat, optdigits.mat, pendigits.mat, pima.mat, satellite.mat, satimage-2.mat, shuttle.mat, smtp.mat, speech.mat, thyroid.mat, vertebral.mat, vowels.mat, wbc.mat, wine.mat, and non ".mat" data: seismic-bumps.arff, yeast.data, yeast.names |

Ensure each of the datasets is put into the correct folder in the `raw_data` folder.

The raw data can then be processed using the `read_raw_write_in_format.py` script.

```
python3 read_raw_write_in_format.py
```

All methods can then be run using a nice CLI:

```
python3 run_all_methods.py
```

Or alternatively, you can add additional arguments to run only subsets. For example, you only want to run the kNN method on the wine dataset:

```
python3 run_all_methods.py --method kNN --dataset wine
```

As noted in the paper, we've inverted the labels for the `skin` and `vertebral` datasets post-hoc. This can be reproduced by executing the following script:

```
python3 invert_labels_calculate_metrics.py
```

Finally, reproducing the figures and analysis from the paper is then easily done using the following command:


```
python3 produce_figures.py
```

## Extending the benchmark
Extending the benchmark is easy!
You'll not need to install all dependencies for this, but a minimal set will do:
```
conda env create -f minimal_environment.yml
```

Then, activate the environment:
```
conda activate OD_benchmark_minimal
```

### Adding datasets
Datasets can be added by ensuring processed datafiles are added to `processed_data` folder in either `.npz` or `.pickle` format. You can look at the `read_raw_write_in_format.py` script for inspiration. Most importantly, data can't contain duplicates, and must include the following attributes: `X` with samples as rows and features as columns, `y`, 1-dimensional with a label `0` for each normal sample, and `1` for each anomaly.

### Adding methods:
Methods are even easier to add, they only need to produce outlier scores according to the PyOD standard. If your implementation follows the Sklearn API, you can simply modify the following example: (also found in `method_example.py`)
```
from preprocess_detect_outliers import preprocess_detect_outliers

from pyod.models.knn import KNN 

#uninstantiated method class
methods = {
        "kNN":KNN
        }

#dict of methods and parameters
method_parameters = {
        "kNN":{"n_neighbors":range(5,31), "method":["mean"]}
        }

preprocess_detect_outliers(methods, method_parameters)
```
If you want to add your method, the class needs to at least possess a `.fit()` function (like the Scikit-learn API), and it must after fitting have an `.decision_scores_` attribute which gives an outlier score for each sample in `X`. According to the PyOD standard, a high outlier score indicates a higher likelihood for a sample to be an outlier.

