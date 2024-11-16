class PipelineEDAGen:
    def __init__(
            self,
            project_name: str,
            run_name: str,
            run_description: str,

            loader,
            harmonizer,
            analyzer,
            **kwargs
    ) -> None:
        
        # Run details
        self.project_name = project_name
        self.run_name = run_name
        self.run_description = run_description

        # Modules
        self.loader = loader
        self.harmonizer = harmonizer
        self.analyzer = analyzer

class PipelineMLGen:
    def __init__(
            self,
            project_name: str,
            run_name: str,
            run_description: str,

            loader,
            harmonizer,
            trainer,
            **kwargs
    ) -> None:
        
        # run details
        self.project_name = project_name
        self.run_name = run_name
        self.run_description = run_description

        # modules
        self.loader = loader
        self.harmonizer = harmonizer
        self.trainer = trainer