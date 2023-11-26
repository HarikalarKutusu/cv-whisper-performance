"""cv-tbox Whisper Performance - Constants"""
###########################################################################
# const.py
#
# Constants for scripts in this repository
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard lib
import os

# Language codes supported by whisper multilingual
from whisper.tokenizer import LANGUAGES

WHISPER_LC: list[str] = sorted(LANGUAGES.keys())

# List of model names to test against, see https://github.com/openai/whisper#available-models-and-languages
# We use multi-lingual models
WHISPER_MODEL_EXT = "pt"
WHISPER_MODELS_ALL: list[str] = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
# VRAM required for model, to calculate max concurrency
WHISPER_MODEL_VRAM: dict[str, int] = {
    "tiny": 1,
    "base": 1,
    "small": 2,
    "medium": 5,
    "large": 10,
    "large-v2": 10,
    "large-v3": 10,
}

# Map from Whisper language to CV lc code
LC_MAPPER: dict[str, str] = {
    "hy": "hy-AM",
    "nn": "nn-NO",
    "pa": "pa-IN",
    "sv": "sv-SE",
    "zh": "zh-CN",
}

# Map from CV lc code to Whisper language
LC_BACK_MAPPER: dict[str, str] = {
    "hy-AM": "hy",
    "nn-NO": "nn",
    "pa-IN": "pa",
    "sv-SE": "sv",
    "zh-CN": "zh",
}

# dir names
DATA_DIR: str = "data"
TEST_SETS_DIR: str = os.path.join(DATA_DIR, "test-sets")
EXPERIMENTS_DIR: str = os.path.join(DATA_DIR, "experiments")
MODELS_DIR: str = os.path.join(DATA_DIR, "models")

# file names
TEST_FN: str = "wtest.tsv"
SUMMARY_FN: str = "wtest_summary.tsv"

#
# DataFrame columns
#

TEST_SET_SUMMARY_COLS: list[str] = [
    "lc",
    "recordings",
    "duration",
    "avg_dur",
    "avg_char_speed",
    "uq_voices",
    "uq_sentences",
]

CV_COLS: list[str] = [
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
]

CV_EXTENDED_COLS: list[str] = [
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
    "norm_sentence",
    "norm_sentence_len",
    "duration",
    "char_speed",
]

TRANSCRIPTION_REC_COLS: list[str] = [
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
    "norm_sentence",
    "norm_sentence_len",
    "duration",
    "char_speed",
    "transcription",
    "norm_transcription",
    "detected_lc",
    "cer",
    "wer",
    "mer",
    "wil",
    "wip",
    "item_inference_duration",
    "item_total_duration",
    "rtf",
]

AGGREGATION_REC_COLS: list[str] = [
    "model",
    "lc",
    "num_sentences",
    "duration",
    "avg_char_speed",
    "inference_duration",
    "total_duration",
    "avg_cer",
    "avg_wer",
    "avg_mer",
    "avg_wil",
    "avg_wip",
    "avg_rtf",
]
