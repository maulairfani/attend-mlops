from zenml import pipeline


@pipeline
def training_pipeline():
    """
    End-to-end face recognition training pipeline.

    Steps:
        1. validate_dataset   — schema & quality checks
        2. preprocess         — resize, normalize, augment
        3. train              — fine-tune face recognition model
        4. evaluate           — compute TAR@FAR and other metrics
        5. register           — push to W&B Model Registry if metrics pass
    """
    pass
