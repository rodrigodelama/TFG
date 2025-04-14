# TFG

This is my Bachelor Thesis: Machine Learning-Based Predictive Modeling of Energy Prices

| Año Académico | Centro      | Plan          | Convocatoria   | Oferta                                                              |
| --------------- | ----------- | ------------- | -------------- | ------------------------------------------------------------------- |
| 2024            | **2** | **447** | **1587** | 62719 - Machine Learning-Based Predictive Modeling of Energy Prices |

## Configure the project environment

Create a venv with the requirements.txt file:

(if on macOS additionally install libomp before installing the requirements: brew install libomp)

```bash
python3 -m venv .tfg_venv
source .tfg_venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt
```

The project is meant to be contained in a virtual environment
To create a virtual environment use: python3 -m venv .tfg_env
To activate the virtual environment use: source .tfg_env/bin/activate
    or in Windows: .tfg_env\Scripts\Activate.ps1
To deactivate the virtual environment use: deactivate

## Steps to execute

### Data downloading

Download the data from the ODBC, either in a .zip format and uncompress, or by configuring with the desired dates, and running the script

```bash
python3 0_download_files.py
```

located in data/data_scrapping

### Build & trim database

To build the initial database from the downloaded files run

```bash
python3 1_db_builder.py
```

To trim the database to the desired target hour run

```bash
python3 2_db_14_and_return_builder.py
```

This will compute the database with exclusively the target hour data, and the return values from that same subset

### Ready to run the TFG.ipynb or separate scripts
