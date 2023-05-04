###########################################################################
# config.py
#
# Your local configuration
# Modify the values to point to respective directories in your system
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################
import os

# Test set size
MAX_DELTA_SIZE: int = 100

# Bias related
UNIQUE_SENTENCES: bool = True
UNIQUE_VOICES: bool = False

# Base location of tsv files for Common Voice v9.0
CV9_DIR: str = os.path.join(
    "C:", os.sep, "GITREPO", "_HK_GITHUB", "_cv_tbox", "cv-tbox-dataset-compiler",
    "data", "voice-corpus", "cv-corpus-9.0-2022-04-27"
    )

# Base location of tsv & audio files for Common Voice v13.0
CV_LATEST_DIR: str = os.path.join(
    "M:", os.sep, "DATASETS", "CV", "cv-corpus-13.0-2023-03-09"
    )

# Base location of whisper model files
WHISPER_MODEL_DIR: str = os.path.join(
    "M:", os.sep, "__STATIC", "_DATASETS", "VOICE", "WHISPER_MODELS"
    )

# List of whisper models to test against
WHISPER_MODELS_TEST: list[str] = [
    "tiny",
    "base",
]
