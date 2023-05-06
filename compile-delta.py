###########################################################################
# compile-delta.py
#
# Compile a delta dataset for whisper supported languages
# - Take difference of validated.tsv in v13.0 and validated.tsv in v9.0 (whisper is based on this)
# - Sort text by desc length
# - Take the longest MAX_DELTA_SIZE (default 1000) unique texts
# - Output to data/cv-delta/[lc]/delta.tsv
# - Copy related audio to data/cv-delta/[lc]/clips/*.mp3
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

import sys, os, shutil, glob, csv
from datetime import datetime, timedelta
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

# MultiProcessing
import multiprocessing as mp
import psutil

# Common Voice Utilities
import cvutils as cvu

# av
import av
import logging
logging.getLogger('libav').setLevel(logging.ERROR) # get rid of warnings


HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Application
import config as conf
import const as c
from library import df_read, df_write, dec2, DeltaResult

# Common Voice Utilities Globals
cv: cvu.CV = cvu.CV()

# Multi Processing Cores
NUM_PROCS: int = psutil.cpu_count(logical=False) - 1

#
# Locale handling (multiprocessed)
#


def handle_locale(lc: str) -> DeltaResult:
    """Handle a single locale (multiprocess)"""
    # precalc paths
    cv9_validated: str = os.path.join(conf.CV9_DIR, lc, "validated.tsv")
    cv_latest_path: str = os.path.join(conf.CV_LATEST_DIR, lc)
    cv_latest_validated: str = os.path.join(cv_latest_path, "validated.tsv")
    cv_latest_audio_path: str = os.path.join(cv_latest_path, "clips")

    # get validated dataframes. cv9 might not exist, latest cv should exist
    if os.path.isdir(os.path.join(cv9_validated, )):
        ext_df: pd.DataFrame = pd.concat([df_read(os.path.join(cv9_validated, "validated.tsv")), df_read(cv_latest_validated)]).drop_duplicates(keep=False)
    else:
        ext_df: pd.DataFrame = df_read(cv_latest_validated)
    
    ext_df.reindex(columns=c.CV_EXTENDED_COLS)

    # diff & extend with new columns

    # Validate & Normalize (we made sure validattor exists)
    v = cvu.Validator(lc)
    ext_df["s_norm"] = ext_df["sentence"].apply(lambda x: v.validate(x))  # normalize
    ext_df["s_norm_len"] = ext_df["s_norm"].str.len()  # normalized length

    # sort by norm sentence length desc (we want long sentences/recordings)
    ext_df.sort_values(["s_norm_len"], ascending=False, inplace=True)

    # Remove duplicate sentences & voices for less bias - if required (set in config.py)
    # These might reduce the size of the test set for
    if conf.UNIQUE_SENTENCES:
        ext_df["s_enum"], s_unique = pd.factorize(
            ext_df["sentence"]
        )  # add an enumaration column for sentences
        num_uq_sentences: int = len(ext_df["s_enum"].unique())
        # conditional bias removal
        if (num_uq_sentences >= conf.MAX_DELTA_SIZE) or (
            num_uq_sentences < conf.MAX_DELTA_SIZE and not conf.UNIQUE_SENTENCES_ONLY_IF_AVAILABLE
        ):
            ext_df.drop_duplicates(subset="s_enum", keep="first", inplace=True)
        ext_df.drop(columns=["s_enum"], inplace=True)

    if conf.UNIQUE_VOICES:
        ext_df["v_enum"], v_unique = pd.factorize(
            ext_df["client_id"]
        )  # add an enumaration column for client_id's
        num_uq_voices: int = len(ext_df["v_enum"].unique())
        # conditional bias removal
        if (num_uq_voices >= conf.MAX_DELTA_SIZE) or (
            num_uq_voices < conf.MAX_DELTA_SIZE and not conf.UNIQUE_VOICES_ONLY_IF_AVAILABLE
        ):
            ext_df.drop_duplicates(subset="v_enum", keep="first", inplace=True)
        ext_df.drop(columns=["v_enum"], inplace=True)

    # Get top
    ext_df = ext_df.head(conf.MAX_DELTA_SIZE)

    # create destination
    dest_path: str = os.path.join(HERE, "data", "cv-delta", lc)
    dest_clips_path: str = os.path.join(dest_path, "clips")
    os.makedirs(dest_clips_path, exist_ok=True)

    # Copy related clip audio files
    clip_names: list[str] = ext_df["path"].to_list()
    for clip_name in clip_names:
        shutil.copy(os.path.join(cv_latest_audio_path, clip_name), dest_clips_path)
    # get audio lengths
    ext_df["a_dur"] = ext_df["path"].apply(
        lambda x: dec2(av.open(os.path.join(cv_latest_audio_path, x)).duration/1000000)
        )

    # save diff.tsv
    df_write(ext_df, os.path.join(dest_path, c.DIFF_FN))

    # Report results
    recs = ext_df.shape[0]
    dur: float = ext_df["a_dur"].sum()
    avg_dur: float = dur/recs
    result: DeltaResult = {
        "lc": lc,
        "recordings": recs,
        "duration": dec2(dur),
        "avg_dur": dec2(avg_dur),
        "uq_voices": len(ext_df["client_id"].unique()),
        "uq_sentences": len(ext_df["sentence"].unique()),
    }
    print(
        f"Finished LC={lc} Added {result['recordings']} recs with {result['duration']} secs from {result['uq_sentences']} sentences and {result['uq_voices']} voices to diff"
    )
    return result

#
# MAIN PROCESS
#


def main() -> None:
    """Main process which loops through whisper languages and handles locales with MP"""

    print(f"==> Whisper supports {len(c.WHISPER_LC)} locales...")

    # Make sure locale exist in latest CV version and has validator
    validator_locales: list[str] = [v.split(os.sep)[-2] for v in cv.validators()]
    lc_list: list[str] = []
    # Whisper might be trained with other sources, so it is OK if not in v9 but it is in latest
    # But we don't want commonvoice-utils unsupported ones
    for lc in c.WHISPER_LC:
        cv_latest_path: str = os.path.join(conf.CV_LATEST_DIR, lc)
        if os.path.isdir(cv_latest_path) and lc in validator_locales:
            lc_list.append(lc)
    print(f"==> Skipping {len(c.WHISPER_LC) - len(lc_list)} locales as they do not exist in CV13 datasets...")

    # Test
    results: list[DeltaResult] = []

    # results.append(handle_locale(lc_list[0])) # single process for testing

    # Multiprocess each locale
    print(f"==> Processing remaining {len(lc_list)} locales...")
    with mp.Pool(NUM_PROCS) as pool:
        results = pool.map(handle_locale, lc_list)

    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.DIFF_SUMMARY_COLS)
    df_write(results_df, os.path.join(HERE, "data", "cv-delta", c.SUMMARY_FN))


# Entry point
if __name__ == "__main__":
    main()
