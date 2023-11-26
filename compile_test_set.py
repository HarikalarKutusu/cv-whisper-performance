"""cv-tbox Whisper Performance - Compile a test set for whisper & CV supported languages"""

###########################################################################
# compile_test_set.py
#
# Compile a test set for whisper & CV supported languages
# - Find the intersection languages for whisper and common-voice latest
# - FOR each locale:
# - Sort text by desc length, do some bias prevention, normalization & some calculations
# - Take the longest MAX_TEST_SIZE (default 100) unique texts
# - Output to data/test-sets/[experiment]/[lc]/wtest.tsv
# - Copy related audio to data/test-sets/[experiment]/[lc]/clips/*.mp3
# - Put aggregated results in data/test-sets/[experiment]
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard lib
import os
import sys
import shutil
import multiprocessing as mp
import logging
import warnings
from typing import List

# External dependencies
import numpy as np
import pandas as pd
import psutil
import cvutils as cvu
import av

# Module
import conf
import const as c
from typedef import TestSetRec
from lib import df_read, df_write, line_count, dec2, lc_back_mapper, lc_mapper

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# get rid of warnings
logging.getLogger("libav").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Common Voice Utilities Globals
cv: cvu.CV = cvu.CV()

# Multi Processing Cores
NUM_PROCS: int = psutil.cpu_count(logical=False)


#
# Locale handling (multiprocessed)
#
def handle_locale(lc: str) -> TestSetRec:
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
        if (num_uq_sentences >= conf.MAX_TEST_SIZE) or (
            num_uq_sentences < conf.MAX_TEST_SIZE and not conf.UNIQUE_SENTENCES_ONLY_IF_AVAILABLE
        ):
            ext_df.drop_duplicates(subset="s_enum", keep="first", inplace=True)
        ext_df.drop(columns=["s_enum"], inplace=True)

    if conf.UNIQUE_VOICES:
        ext_df["v_enum"], v_unique = pd.factorize(ext_df["client_id"])  # add an enumaration column for client_id's
        num_uq_voices: int = len(ext_df["v_enum"].unique())
        # conditional bias removal
        if (num_uq_voices >= conf.MAX_TEST_SIZE) or (
            num_uq_voices < conf.MAX_TEST_SIZE and not conf.UNIQUE_VOICES_ONLY_IF_AVAILABLE
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
        is_ok, res = v.normalise(row["sentence"])
        if is_ok:
            ext_df.at[inx, "norm_sentence"] = res
            ext_df.at[inx, "norm_sentence_len"] = len(res)
            cnt += 1
        else:
            ext_df.at[inx, "norm_sentence"] = np.nan
        if cnt > conf.MAX_TEST_SIZE:
            break  # we've got enough samples

    ext_df.dropna(subset=["norm_sentence"], inplace=True)  # drop invalidated ones

    # Get top N
    ext_df = ext_df.head(conf.MAX_TEST_SIZE).reset_index(drop=True)

    # create destination
    dest_path: str = os.path.join(HERE, c.TEST_SETS_DIR, conf.TEST_SET, lc)
    dest_clips_path: str = os.path.join(dest_path, "clips")
    os.makedirs(dest_clips_path, exist_ok=True)

    # Copy related clip audio files
    clip_names: List[str] = ext_df["path"].to_list()
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
    df_write(ext_df, os.path.join(dest_path, c.TEST_FN))

    # Report results
    recs: int = ext_df.shape[0]
    dur: float = ext_df["duration"].sum()
    avg_dur: float = dur / recs
    avg_char_speed: float = 1000 * dur / ext_df["norm_sentence_len"].sum()
    back_lc: str = lc_back_mapper(lc)
    result: TestSetRec = {
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
    """Main process forms a applicable LC list, and loops tests with MP"""

    # print(f"==> Whisper supports {len(c.WHISPER_LC)} locales...", c.WHISPER_LC)
    print(f"==> Whisper supports {len(c.WHISPER_LC)} locales: {c.WHISPER_LC}")
    # re-map whisper languages to CV codes
    wlc_list: List[str] = [lc_mapper(lc) for lc in c.WHISPER_LC]
    print(f"+++ Whisper mapped to CV codes: {wlc_list}")

    # Make sure locale exist in latest CV version and has validator
    validator_locales: List[str] = sorted([v.split(os.sep)[-2] for v in cv.validators()])
    # print(f"==> Validator supports {len(validator_locales)} locales...", validator_locales)
    print(f"+++ Validator supports {len(validator_locales)} locales: {validator_locales}")

    # get the intersection
    test_lc_list: List[str] = sorted(list(set(wlc_list) & set(validator_locales)))
    print(f"==> Intersection: {len(test_lc_list)} locales: {test_lc_list}")

    # Eliminate non-existing locales
    eliminated_lc: List[str] = []
    for lc in test_lc_list:
        if not os.path.isdir(os.path.join(conf.CV_LATEST_DIR, lc)) or not os.path.isfile(
            os.path.join(conf.CV_LATEST_DIR, lc, "validated.tsv")
        ):
            eliminated_lc.append(lc)
    print(f"--- Non-existing: {len(eliminated_lc)} locales... ({eliminated_lc})")
    test_lc_list = sorted(list(set(test_lc_list) - set(eliminated_lc)))
    print(f"==> Remaining: {len(test_lc_list)} locales: {test_lc_list}")

    # Eliminate locales with low resource
    eliminated_lc = []
    for lc in test_lc_list:
        df: pd.DataFrame = df_read(os.path.join(conf.CV_LATEST_DIR, lc, "validated.tsv"))
        if df.shape[0] < conf.MIN_TEST_SIZE:
            eliminated_lc.append(lc)
    print(f"--- Low-resource: {len(eliminated_lc)} locales... ({eliminated_lc})")
    test_lc_list = sorted(list(set(test_lc_list) - set(eliminated_lc)))
    print(f"==> FINAL: {len(test_lc_list)} locales... ({test_lc_list})")

    # Multiprocess each locale
    result_list: list[TestSetRec] = []
    with mp.Pool(NUM_PROCS) as pool:
        result_list = pool.map(handle_locale, test_lc_list)

    results_df: pd.DataFrame = pd.DataFrame.from_records(result_list, columns=c.TEST_SET_SUMMARY_COLS)
    p: str = os.path.join(HERE, c.TEST_SETS_DIR, conf.TEST_SET)
    os.makedirs(p, exist_ok=True)
    fp: str = os.path.join(p, c.SUMMARY_FN)
    df_write(results_df, fp)


# Entry point
if __name__ == "__main__":
    main()
