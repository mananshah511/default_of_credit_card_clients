from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                   ["is_ingested","message","train_file_path","test_file_path"])


DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated","message","schema_file_path","reprot_file_path"]
                                    )

DataTransformArtifact = namedtuple("DataTransformArtifact",
                                   ["is_transform","message","transform_train_dir","transform_test_dir",
                                    "preprocessed_dir","cluster_model_dir"])
