import sys, os, shutil, glob, csv
import pandas as pd
import whisper
from typing import TypedDict

import config as conf

def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f'FATAL: File {fpath} cannot be located!')
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines='skip',
        quotechar='"',
        quoting=csv.QUOTE_NONE,
        dtype={'ver': str}
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(fpath, header=True, index=False, encoding="utf-8", sep='\t', escapechar='\\', quoting=csv.QUOTE_NONE)
    if conf.VERBOSE:
        print(f'Generated: {fpath} Records={df.shape[0]}')
    return True

def decN(x: float, n: int) -> float:
    N = 10**n
    return int(N * x)/N

def dec2(x: float) -> float:
    return int(100 * x)/100

#
# Type definitions
#
class DeltaResult(TypedDict):
    lc: str
    recordings: int
    duration: float
    avg_dur: float
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
    gender:	str
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
    gender:	str
    accents: str
    variant: str
    locale: str
    segment: str
    s_norm: str             # Normalized sentence
    s_norm_len: str         # Length
    a_dur: str              # Audio duration

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
    gender:	str
    accents: str
    variant: str
    locale: str
    segment: str
    transcription: str
    detected_lc: str
    cer: float
    wer: float
    mer: float
    wil: float
    wip: float
    item_inference_duration: float
    item_total_duration: float

class AggregationRec(TypedDict):
    model: str
    lc: str
    num_sentences: int
    inference_duration: float
    total_duration: float
    avg_cer: float
    avg_wer: float
    avg_mer: float
    avg_wil: float
    avg_wip: float
