## System Requirements

### Hardware requirements

The program requires a standard computer with at least 8GB RAM to support the in-memory computation.

### Software requirements

- Operating System: Linux Ubuntu 16.04
- Python version: 3.7.3
- Python dependencies: numpy(v1.17.2), pandas(v0.25.1), scikit-learn(v0.21.3), scipy(v1.3.1)
- R version: 3.4.4
- R dependencies: MALDIquantForeign(v0.12)

## Installation Guide

After unzipping the code, install the necessary dependency packages

```
pip install -r requirements.txt
```

It will take several minutes to install, depends on the network.

## Directory Structure

```
├── README.md
├── Tools
│   ├── Assessment.py
│   ├── DataTools.py
│   └── FileTools.py
├── mzml2csv
│   └── initData.py
├── preprocess
│   └── dataPreprocess.py
├── model
│   └── model.py
├── plot
│   ├── cm.py
│   └── roc.py
└── requirements.txt
```

The `Tools` folder stores tool files, including functions related to file reading and writing, data organization, and metric evaluation. The file `initData.py` in the `mzml2csv` folder is used to reformat the original data, and the mzml format file is converted to csv format by calling the program in R. The file `dataPreprocess` in `preprocess` folder is used to preprocess the converted data. The file `model.py` in `model` folder is the main file of the method and contains the key steps of model training. The `plot ` folder contains two files for drawing confusion matrix and ROC curves. 

## License

Apache 2.0 License.