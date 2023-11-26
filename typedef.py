"""cv-tbox Whisper Performance - Type Definitions"""
###########################################################################
# typedef.py
#
# Type definitions for this module
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard lib
from typing import TypedDict

# External dependencies
import whisper


#
# Type definitions
#
class TestSetRec(TypedDict):
    """[TODO]"""

    lc: str
    recordings: int
    duration: float
    avg_dur: float
    avg_char_speed: float
    uq_voices: int
    uq_sentences: int


class HandleLocaleProps(TypedDict):
    """[TODO]"""

    model_name: str
    model: whisper.Whisper
    diff_path: str


class CommonVoiceRec(TypedDict):
    """[TODO]"""

    client_id: str
    path: str
    sentence: str
    up_votes: int
    down_votes: int
    down_votes: int
    age: str
    gender: str
    accents: str
    variant: str
    locale: str
    segment: str


class CommonVoiceExtended(TypedDict):
    """[TODO]"""

    client_id: str
    path: str
    sentence: str
    up_votes: int
    down_votes: int
    down_votes: int
    age: str
    gender: str
    accents: str
    variant: str
    locale: str
    segment: str

    norm_sentence: str  # Normalized sentence
    norm_sentence_len: str  # Length
    duration: str  # Audio duration
    char_speed: float


class WhisperTranscriptionResult(TypedDict):
    """[TODO]"""

    text: str
    segments: dict
    language: str


class TranscriptionRec(TypedDict):
    """[TODO]"""

    client_id: str
    path: str
    sentence: str
    up_votes: int
    down_votes: int
    down_votes: int
    age: str
    gender: str
    accents: str
    variant: str
    locale: str
    segment: str

    norm_sentence: str  # Normalized sentence
    norm_sentence_len: str  # Length
    duration: str  # Audio duration
    char_speed: float

    transcription: str
    norm_transcription: str
    detected_lc: str

    cer: float
    wer: float
    mer: float
    wil: float
    wip: float

    item_inference_duration: float
    item_total_duration: float

    rtf: float


class AggregationRec(TypedDict):
    """[TODO]"""

    model: str
    lc: str
    num_sentences: int
    duration: float
    avg_char_speed: float
    inference_duration: float
    total_duration: float
    avg_cer: float
    avg_wer: float
    avg_mer: float
    avg_wil: float
    avg_wip: float
    avg_rtf: float
