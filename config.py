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
# Common Voice Directories, needed for compile-test-set.py script
#

# Base location of tsv & audio files for latest Common Voice (currently v13.0)
CV_LATEST_DIR: str = os.path.join("M:", os.sep, "DATASETS", "CV", "cv-corpus-13.0-2023-03-09")

#
# Whisper model related settings
#

# This is the subdir where the model resides under data/models
# You can put custom models in other subdirs and look for them
# E.g. you can create small-en, small-de etc into "test" sub dir and set WHISPER_MODELS_DIR to "test"
# This time, you should use 
WHISPER_MODELS_DIR: str = os.path.join("M:", os.sep, "__STATIC", "MODELS", "VOICE", "WHISPER", "default")

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
# Subdir in data/test
TEST_SET: str = "longest"

# Subdir in data/results
EXPERIMENT: str = "baseline-gpu"

# compile-test-set.py - Test set maximumn size
MAX_TEST_SIZE: int = 100

# compile-test-set.py - Bias removal related
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
# Your GPU number if you have multiple GPU's. You can use the hw-info script to find the number.
GPU: int = 0
