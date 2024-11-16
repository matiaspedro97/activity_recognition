import os

from src.pipeline.pipe import PipelineEDARunner, PipelineMLRunner
from src.config import pipeline_dir

# pipeline EDA config
config_path = os.path.join(pipeline_dir, 'pipe_ml.json')

# instantiate pipeline runner
pipe_ml = PipelineMLRunner(config_path=config_path)

# run pipeline
metrics = pipe_ml.run()