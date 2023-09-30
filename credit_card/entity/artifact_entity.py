from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                   ["is_ingested","message","train_file_path","test_file_path"])


DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["is_validated","message","schema_file_path","reprot_file_path"]
                                    )