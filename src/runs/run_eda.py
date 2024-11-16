import os

from src.pipeline.pipe import PipelineEDARunner, PipelineMLRunner
from src.config import pipeline_dir

# pipeline EDA config
config_path = os.path.join(pipeline_dir, 'pipe_eda.json')

# instantiate pipeline runner
pipe_eda = PipelineEDARunner(config_path=config_path)

# run pipeline
fig, (tests, p_values, corr_pairs) = pipe_eda.run()

