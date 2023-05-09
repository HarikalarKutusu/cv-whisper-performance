###########################################################################
# config.py
#
# Your local configuration
# Modify the values to point to respective directories in your system
# Modify the test related values (size and bias related settings)
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################
import os

#
# Common Voice Directories, needed for compile-delta.py script
#

# Base location of tsv files for Common Voice v9.0
# We only need the validated.tsv files in the locale directories.
# As whisper is trained on CV v9.0, we need to exclude them from tests to prevent bias.
# You can get the TSV only files from Common Voice Dataset Analyzer application:
# - Visit https://analyzer.cv-toolbox.web.tr/
# - From the dataset browser select the locale's v9.0 dataset
# - Use the download button labeled "s1" to download
# - Expand it under "cv-corpus-9.0-2022-04-27/<lc>"
CV9_DIR: str = os.path.join(
    "C:",
    os.sep,
    "GITREPO",
    "_HK_GITHUB",
    "_cv_tbox",
    "cv-tbox-dataset-compiler",
    "data",
    "voice-corpus",
    "cv-corpus-9.0-2022-04-27",
)

# Base location of tsv & audio files for latest Common Voice (currently v13.0)
CV_LATEST_DIR: str = os.path.join("M:", os.sep, "DATASETS", "CV", "cv-corpus-13.0-2023-03-09")

#
# Whisper model related settings
#

# This is the subdir where the model resides under data/models
# You can put custom models in other subdirs and look for them
# E.g. you can create small-en, small-de etc into "test" sub dir and set WHISPER_MODELS_DIR to "test"
# This time, you should use 
WHISPER_MODELS_DIR: str = "default"

# List of whisper models to test against
# Each test runs for 30 min - 3 hours on a 6*2 core CPU & rtx-3090
# So you might want to run these one by one, or multiple for overnight
# If you are only interested for a specific one, just uncomment it
WHISPER_MODELS_TO_TEST: list[str] = [
    "tiny",
    # "base",
    # "small",
    # "medium",
    # "large-v1",
    # "large-v2",
]


#
# Test settings
#

# Subdir in data/results
EXPERIMENT: str = "baseline-gpu"

# compile-delta.py - Test set maximumn size
MAX_DELTA_SIZE: int = 100

# compile-delta.py - Bias removal related
UNIQUE_SENTENCES: bool = True
UNIQUE_SENTENCES_ONLY_IF_AVAILABLE: bool = True
UNIQUE_VOICES: bool = True
UNIQUE_VOICES_ONLY_IF_AVAILABLE: bool = True

# Forced include languages to only process for them (keep empty for all)
INCLUDE_LANGUAGES: list[str] = [
    "tr",
]

# Force exclude languages (if )
EXCLUDED_LANGUAGES: list[str] = [
    "is",  # no validated data in v13.0
]

#
# Program parameters
#

VERBOSE: bool = False               # If true, more will be printed on the console
FAIL_ON_NOT_FOUND: bool = True      # Fail process if dataset is not found

#
# GPU
#

# For GPU usage, set to True, for CPU set to False
USE_GPU: bool = True
# Your VRAM size in GB, we predict max processes to prevent memory full errors
GPU: int = 0
