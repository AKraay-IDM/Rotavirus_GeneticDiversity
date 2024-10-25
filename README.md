# RotaABM

This repository contains code for the rotavirus genetic diversity project. The model is calibrated to data from Bangladesh, focusing on the pre-vaccine period.

## Installation

`pip install -r requirements.txt`

This does _not_ install the code as an importable module; just run `rotaABM.py` for that (see below).

## Running the code

Call `python rotaABM.py` which writes results to the `results` folder. File names are set based on parameter values. Run `data_cleaning.py` to parse the output files into summary statistics.