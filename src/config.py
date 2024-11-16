import os

from os import getenv
from os.path import dirname, join
import dotenv

# project directory
project_dir = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])

# Load the environment variables from the `.env` file, overriding any system environment variables
env_path = join(project_dir, '.env')
dotenv.load_dotenv(env_path, override=True)

# datasets
data_dir = os.path.join(project_dir, 'data')
data_raw_dir = os.path.join(data_dir, 'raw')
data_proc_dir = os.path.join(data_dir, 'processed')

# configs dirs
config_dir = os.path.join(project_dir, 'configs')

pipeline_dir = os.path.join(config_dir, 'pipeline')
scaler_dir = os.path.join(config_dir, 'scaler')
selector_dir = os.path.join(config_dir, 'selector')
classifier_dir = os.path.join(config_dir, 'classifier')

# config paths
scaler_path = os.path.join(scaler_dir, 'scalers.json')
selector_path = os.path.join(selector_dir, 'selectors.json')
classifier_path = os.path.join(classifier_dir, 'classifiers.json')

# datasets
har_dataset_path = os.path.join(data_proc_dir, 'dataset_assignment.csv')