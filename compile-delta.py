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

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Application
import config as conf
import const as c
from library import df_read, df_write

# Multi Processing Cores
NUM_PROCS: int = psutil.cpu_count(logical=False) - 1

#
# Locale handling (multiprocessed)
#

def handle_locale(lc: str) -> None:
    """Handle a single locale (multiprocess)"""
    # precalc paths
    cv9_path = os.path.join(conf.CV9_DIR, lc)
    cv_latest_path = os.path.join(conf.CV_LATEST_DIR, lc)
    cv_latest_audio_path = os.path.join(cv_latest_path, "clips")

    # get validated dataframes
    validated_v9_df: pd.DataFrame = df_read(os.path.join(cv9_path, "validated.tsv"))
    validated_latest_df: pd.DataFrame = df_read(os.path.join(cv_latest_path, "validated.tsv"))

    # diff
    diff_df: pd.DataFrame = pd.concat([validated_v9_df, validated_latest_df]).drop_duplicates(keep=False)

    # Get sentence lengths, sort by length desc (we want long recordings)
    diff_df["s_len"] = diff_df["sentence"].str.len()
    diff_df.sort_values(["s_len"], ascending=False, inplace=True)
    diff_df.drop(columns=["s_len"], inplace=True)

    # Remove duplicate sentences & voices for less bias - if required (set in config.py)
    # These might reduce the size of the test set for 
    if conf.UNIQUE_SENTENCES:
        diff_df['s_enum'], s_unique = pd.factorize(diff_df['sentence']) # add an enumaration column for client_id's
        num_uq_sentences = len(diff_df["s_enum"].unique())
        # conditional bias removal
        if (num_uq_sentences >= conf.MAX_DELTA_SIZE) or (num_uq_sentences < conf.MAX_DELTA_SIZE and not conf.UNIQUE_SENTENCES_ONLY_IF_AVAILABLE):
            diff_df.drop_duplicates(subset="s_enum", keep="first", inplace=True)
        diff_df.drop(columns=["s_enum"], inplace=True)

    if conf.UNIQUE_VOICES:
        diff_df['v_enum'], v_unique = pd.factorize(diff_df['client_id']) # add an enumaration column for client_id's
        num_uq_voices = len(diff_df["v_enum"].unique())
        # conditional bias removal
        if (num_uq_voices >= conf.MAX_DELTA_SIZE) or (num_uq_voices < conf.MAX_DELTA_SIZE and not conf.UNIQUE_VOICES_ONLY_IF_AVAILABLE):
            diff_df.drop_duplicates(subset="v_enum", keep="first", inplace=True)
        diff_df.drop(columns=["v_enum"], inplace=True)

    # Get top
    diff_df = diff_df.head(conf.MAX_DELTA_SIZE)

    # create destination
    dest_path = os.path.join(HERE, "data", "cv-delta", lc)
    dest_clips_path = os.path.join(dest_path, "clips")
    os.makedirs(dest_clips_path, exist_ok=True)

    # save diff.tsv
    df_write(diff_df, os.path.join(dest_path, "diff.tsv"))

    # Copy related clip audio files
    clip_names: list[str] = diff_df["path"].to_list()
    for clip_name in clip_names:
        shutil.copy(os.path.join(cv_latest_audio_path, clip_name), dest_clips_path)

    # Report results
    result = {
        "lc": lc,
        "recordings": diff_df.shape[0],
        "uq_voices": len(diff_df["client_id"].unique()),
        "uq_sentences": len(diff_df["sentence"].unique())
    }
    print(f"Finished LC={lc} Added {result['recordings']} recordings from {result['uq_sentences']} sentences and {result['uq_voices']} voices to diff")
    return result

#
# MAIN PROCESS
#

def main() -> None:
    """Main process which loops through whisper languages and handles locales with MP"""

    print(f'==> Whisper supports {len(c.WHISPER_LC)} locales...')
    # Make sure all exist
    lc_list: list[str] = []
    for lc in c.WHISPER_LC:
        cv9_path = os.path.join(conf.CV9_DIR, lc)
        cv_latest_path = os.path.join(conf.CV_LATEST_DIR, lc)
        if os.path.isdir(cv9_path) and os.path.isdir(cv_latest_path):  # both should exist
            lc_list.append(lc)
    print(f'==> Skipping {len(c.WHISPER_LC) - len(lc_list)} locales as they do not exist in CV9 & CV13 datasets...')

    # Test
    results: list[dict[str, int]] = []
    # results.append(handle_locale(lc_list[0]))

    # Multiprocess each locale
    print(f'==> Processing remaining {len(lc_list)} locales...')
    with mp.Pool(NUM_PROCS) as pool:
        results = pool.map(handle_locale, lc_list)

    results_df = pd.DataFrame.from_records(results, columns=["lc", "recordings", "uq_voices", "uq_sentences"])
    # print(results)
    # print(results_df)
    df_write(results_df, os.path.join(HERE, "data", "cv-delta", "summary.tsv"))

# Entry point
if __name__ == '__main__':
    main()
