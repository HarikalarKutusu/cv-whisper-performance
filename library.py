import sys, os, shutil, glob, csv
import pandas as pd
import whisper
from typing import TypedDict

import const as c
import config as conf


def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
        dtype={"ver": str},
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(fpath, header=True, index=False, encoding="utf-8", sep="\t", escapechar="\\", quoting=csv.QUOTE_NONE)
    if conf.VERBOSE:
        print(f"Generated: {fpath} Records={df.shape[0]}")
    return True


def lc_mapper(lc: str) -> str:
    if lc in c.LC_MAPPER.keys():
        return c.LC_MAPPER[lc]
    else:
        return lc


def lc_back_mapper(lc: str) -> str:
    if lc in c.LC_BACK_MAPPER.keys():
        return c.LC_BACK_MAPPER[lc]
    else:
        return lc


def dec2(x: float) -> float:
    return round(x, 2)

def dec6(x: float) -> float:
    return round(x, 6)

def bytes2gb(mem: int) -> float:
    return dec2(mem / (1024 * 1024 * 1024))

#
# Type definitions
#
class DeltaResult(TypedDict):
    lc: str
    recordings: int
    duration: float
    avg_dur: float
    avg_char_speed: float
    uq_voices: int
    uq_sentences: int


class HandleLocaleProps(TypedDict):
    model_name: str
    model: whisper.Whisper
    diff_path: str


class CommonVoiceRec(TypedDict):
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
    text: str
    segments: dict
    language: str


class TranscriptionRec(TypedDict):
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
