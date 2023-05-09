###########################################################################
# compile-delta.py
#
# Compile a delta dataset for whisper supported languages
# - Take difference of validated.tsv in v13.0 and validated.tsv in v9.0 (whisper is based on this)
# - Sort text by desc length, do some bias prevention, normalization & some calculations
# - Take the longest MAX_DELTA_SIZE (default 100) unique texts
# - Output to data/cv-delta/[lc]/diff.tsv
# - Copy related audio to data/cv-delta/[lc]/clips/*.mp3
# - Put aggregated results in data/cv-delta
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

logging.getLogger("libav").setLevel(logging.ERROR)  # get rid of warnings


HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# Application
import config as conf
import const as c
from library import df_read, df_write, dec2, DeltaResult, lc_back_mapper, lc_mapper

# Common Voice Utilities Globals
cv: cvu.CV = cvu.CV()

# Multi Processing Cores
NUM_PROCS: int = psutil.cpu_count(logical=False)


#
# Locale handling (multiprocessed)
#
def handle_locale(lc: str) -> DeltaResult:
    """Handle a single locale (multiprocess). LC is in CV terms."""
    # print("Started:", lc)
    # precalc paths
    _cv_latest_path: str = os.path.join(conf.CV_LATEST_DIR, lc)
    cv_latest_validated: str = os.path.join(_cv_latest_path, "validated.tsv")
    cv_latest_audio_path: str = os.path.join(_cv_latest_path, "clips")

    # get validated dataframes from lates CV release
    ext_df: pd.DataFrame = df_read(cv_latest_validated)

    # Remove duplicate sentences & voices for less bias - if required (set in config.py)
    # These might reduce the size of the test set for
    if conf.UNIQUE_SENTENCES:
        ext_df["s_enum"], s_unique = pd.factorize(ext_df["sentence"])  # add an enumaration column for sentences
        num_uq_sentences: int = len(ext_df["s_enum"].unique())
        # conditional bias removal
        if (num_uq_sentences >= conf.MAX_DELTA_SIZE) or (
            num_uq_sentences < conf.MAX_DELTA_SIZE and not conf.UNIQUE_SENTENCES_ONLY_IF_AVAILABLE
        ):
            ext_df.drop_duplicates(subset="s_enum", keep="first", inplace=True)
        ext_df.drop(columns=["s_enum"], inplace=True)

    if conf.UNIQUE_VOICES:
        ext_df["v_enum"], v_unique = pd.factorize(ext_df["client_id"])  # add an enumaration column for client_id's
        num_uq_voices: int = len(ext_df["v_enum"].unique())
        # conditional bias removal
        if (num_uq_voices >= conf.MAX_DELTA_SIZE) or (
            num_uq_voices < conf.MAX_DELTA_SIZE and not conf.UNIQUE_VOICES_ONLY_IF_AVAILABLE
        ):
            ext_df.drop_duplicates(subset="v_enum", keep="first", inplace=True)
        ext_df.drop(columns=["v_enum"], inplace=True)

    # sort desc by sentence length first (to prevent full validation in large datasets)
    ext_df["s_len"] = ext_df["sentence"].str.len()  # normalized length
    ext_df.sort_values(["s_len"], ascending=False, inplace=True)
    ext_df.drop(columns=["s_len"], inplace=True)

    # extend with new columns
    ext_df.reindex(columns=c.CV_EXTENDED_COLS).reset_index(drop=True)

    # Validate & Normalize (we made sure validator exists for this locale)
    v = cvu.Validator(lc)
    cnt: int = 0
    for inx, row in ext_df.iterrows():
        isOK, res = v.normalise(row["sentence"])
        if isOK:
            ext_df.at[inx, "norm_sentence"] = res
            ext_df.at[inx, "norm_sentence_len"] = len(res)
            cnt += 1
        else:
            ext_df.at[inx, "norm_sentence"] = np.nan
        if cnt > conf.MAX_DELTA_SIZE:
            break  # we've got enough samples

    ext_df.dropna(subset=["norm_sentence"], inplace=True)  # drop invalidated ones

    # Get top N
    ext_df = ext_df.head(conf.MAX_DELTA_SIZE).reset_index(drop=True)

    # create destination
    dest_path: str = os.path.join(HERE, "data", "test", conf.EXPERIMENT, lc)
    dest_clips_path: str = os.path.join(dest_path, "clips")
    os.makedirs(dest_clips_path, exist_ok=True)

    # Copy related clip audio files
    clip_names: list[str] = ext_df["path"].to_list()
    for clip_name in clip_names:
        shutil.copy(os.path.join(cv_latest_audio_path, clip_name), dest_clips_path)
    # get audio lengths
    ext_df["duration"] = ext_df["path"].apply(
        lambda x: dec2(av.open(os.path.join(cv_latest_audio_path, x)).duration / 1000000)
    )
    # calc record char speed
    ext_df["char_speed"] = 1000 * ext_df["duration"] / ext_df["norm_sentence_len"]
    ext_df["char_speed"] = ext_df["char_speed"].apply(lambda x: dec2(x))
    ext_df["duration"] = ext_df["duration"].apply(lambda x: dec2(x))

    # save diff.tsv
    df_write(ext_df, os.path.join(dest_path, c.DIFF_FN))

    # Report results
    recs: int = ext_df.shape[0]
    dur: float = ext_df["duration"].sum()
    avg_dur: float = dur / recs
    avg_char_speed: float = 1000 * dur / ext_df["norm_sentence_len"].sum()
    back_lc: str = lc_back_mapper(lc)
    result: DeltaResult = {
        "lc": lc if lc == back_lc else f"{lc} ({back_lc})",
        "recordings": recs,
        "duration": dec2(dur),
        "avg_dur": dec2(avg_dur),
        "avg_char_speed": dec2(avg_char_speed),
        "uq_voices": len(ext_df["client_id"].unique()),
        "uq_sentences": len(ext_df["sentence"].unique()),
    }
    print("Finished", result)
    return result


#
# MAIN PROCESS
#


def main() -> None:
    """Main process which loops through whisper languages and handles locales with MP"""

    # print(f"==> Whisper supports {len(c.WHISPER_LC)} locales...", c.WHISPER_LC)
    print(f"==> Whisper supports {len(c.WHISPER_LC)} locales...")

    # Make sure locale exist in latest CV version and has validator
    validator_locales: list[str] = [v.split(os.sep)[-2] for v in cv.validators()]
    # print(f"==> Validator supports {len(validator_locales)} locales...", validator_locales)
    print(f"==> Validator supports {len(validator_locales)} locales...")
    lc_list: list[str] = []
    # Whisper might be trained with other sources, so it is OK if not in v9 but it is in latest
    # But we don't want commonvoice-utils unsupported ones
    for lc in c.WHISPER_LC:
        mapped_lc: str = lc_mapper(lc)
        cv_latest_path: str = os.path.join(conf.CV_LATEST_DIR, mapped_lc)
        if (
            os.path.isdir(cv_latest_path)
            and (mapped_lc in validator_locales)
            and not (mapped_lc in conf.EXCLUDED_LANGUAGES)
        ):
            lc_list.append(mapped_lc)
        else:
            print("Skipped:", lc, mapped_lc)

    print(f"==> Skipped {len(c.WHISPER_LC) - len(lc_list)} locales as they do not exist in CV/Validator...")

    # Test
    result_list: list[DeltaResult] = []

    # Multiprocess each locale
    print(f"==> Processing remaining {len(lc_list)} locales...")
    with mp.Pool(NUM_PROCS) as pool:
        result_list = pool.map(handle_locale, lc_list)

    results_df: pd.DataFrame = pd.DataFrame.from_records(result_list, columns=c.DIFF_SUMMARY_COLS)
    df_write(results_df, os.path.join(HERE, "data", "cv-delta", c.SUMMARY_FN))


# Entry point
if __name__ == "__main__":
    main()
