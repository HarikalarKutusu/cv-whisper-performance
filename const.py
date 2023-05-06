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

# Language codes supported by whisper multilingual
WHISPER_LC: list[str] = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "nb",
    "ne",
    "nl",
    "nn",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
]

# List of model names to test against, see https://github.com/openai/whisper#available-models-and-languages
# We use multi-lingual models
WHISPER_MODEL_EXT = "pt"
WHISPER_MODELS_ALL: list[str] = ["tiny", "base", "small", "medium", "large-v2"]
# VRAM required for model, to calculate max concurrency
WHISPER_MODEL_VRAM: dict[str, int] = {"tiny": 1, "base": 1, "small": 2, "medium": 5, "large-v2": 10}

# files
DIFF_FN = "diff.tsv"
SUMMARY_FN = "diff_summary.tsv"

#
# DataFrame columns
#
DIFF_SUMMARY_COLS: list[str] = ["lc", "recordings", "duration", "avg_dur", "uq_voices", "uq_sentences"]

CV_COLS: list[str] = [
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
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
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
    "s_norm",
    "s_norm_len",
    "a_dur",
]

TRANSCRIPTION_REC_COLS: list[str] = [
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
    "transcription",
    "detected_lc",
    "cer",
    "wer",
    "mer",
    "wil",
    "wip",
    "item_inference_duration",
    "item_total_duration",
]

AGGREGATION_REC_COLS: list[str] = [
    "model",
    "lc",
    "num_sentences",
    "inference_duration",
    "total_duration",
    "avg_cer",
    "avg_wer",
    "avg_mer",
    "avg_wil",
    "avg_wip",
]
