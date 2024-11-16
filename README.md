activity_recognition
==============================

Machine Learning Challenge for Human Activity Recognition

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


## Setup Environment
To setup the environment, you should build two distinct virtual environment: (1) conda environment (with basic ML project packages) and (2) virtualenv (with essencial packages to run the ML pipelines)

#### 1. Make sure to install Anaconda or Miniconda

#### 2. Setup the Conda enviroment.
```bash
conda create env -file environment.yml
```

#### 3. Activate the new environment
```bash
conda deactivate

conda activate python3.8
```

#### 4. Build the dev virtualenv on top of the conda environment
```bash
virtualenv .venv-dev
```

#### 5. Activate the dev virtualenv

```bash
. .venv-dev/bin/activate  # linux
```
OR 
```bash
.venv-dev/Scripts/activate  # windows
```

#### 6. Install the dependencies
With both environments activated, install hard dependencies
```bash
pip install -r requirements/requirements-dev.txt  # windows
```

#### 7. You are now able to run the scripts

```bash
# exploratory data analysis from config file settings
python src/runs/run_eda.py

# ML experiments from config file settings
python src/runs/run_ml_training.py
```

## Reports
Please check the Reports folder to see some of the obtained results: [Reports](reports/)

## Configuration
Please check the pipeline configuration files. You'll need to define one to run the experiment scripts: [Configs](configs/)

## Contributions
Contributions are welcomed! If you want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request detailing your changes.

## License

This project is licensed under the [MIT License](LICENSE).  

## Contact
For any questions or inquiries, please contact matiaspedro97@gmail.com

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


