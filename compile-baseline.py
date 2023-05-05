###########################################################################
# compile-baseline.py
#
# Tests the compiled test set against the selected default whisper models
# and logs the results in detail and as aggregated values
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

# whisper
import whisper

# Try to get rid of deprecation warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# jiwer
import jiwer

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
from library import (
    AggregationRec,
    HandleLocaleProps,
    WhisperTranscriptionResult,
    df_read,
    df_write,
    CommonVoiceRec,
    TranscriptionRec,
)

# Multi Processing Cores
NUM_PROCS: int = psutil.cpu_count(logical=False) - 1

#
# Locale handling (multiprocessed)
#


def handle_locale(model_name: str, diff_path: str) -> AggregationRec:
    """Handle a single locale (multiprocess)"""
    start_locale: datetime = datetime.now()
    # model_name: str = props["model_name"]
    # model: whisper.Whisper = props["model"]
    # diff_path: str = props["diff_path"]
    model_dir: str = os.path.join(HERE, "data", "models", "default")
    model: whisper.Whisper = whisper.load_model(name=model_name, download_root=model_dir, in_memory=True)

    # get dir
    locale_path: str = os.path.split(diff_path)[0]
    lc_tsv: str = os.path.split(diff_path)[1]
    lc: str = locale_path.split(os.sep)[-1]

    # get diff dataframe as result and expand it with new columns
    source_df: pd.DataFrame = df_read(diff_path)

    results: list[TranscriptionRec] = []
    # Loop through each record
    for inx, row in source_df.iterrows():
        start_row: datetime = datetime.now()
        result: TranscriptionRec = row.to_dict()  # type: ignore
        audio_path: str = os.path.join(locale_path, "clips", row["path"])
        start_transcription: datetime = datetime.now()
        transcription_result: WhisperTranscriptionResult = whisper.transcribe(model, audio_path)  # type: ignore
        result["item_inference_duration"] = (datetime.now() - start_transcription).total_seconds()
        result["transcription"] = transcription_result["text"].strip()
        # result["segments"] = transcription_result["segments"]
        result["detected_lc"] = transcription_result["language"]
        j_word: jiwer.WordOutput = jiwer.process_words(reference=row["sentence"], hypothesis=result["transcription"])
        j_char: jiwer.CharacterOutput = jiwer.process_characters(
            reference=row["sentence"], hypothesis=result["transcription"]
        )
        result["cer"] = j_char.cer
        result["wer"] = j_word.wer
        result["mer"] = j_word.mer
        result["wil"] = j_word.wil
        result["wip"] = j_word.wip
        result["item_total_duration"] = (datetime.now() - start_row).total_seconds()
        results.append(result)  # type: ignore

    # save locale results for this model
    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.TRANSCRIPTION_REC_COLS)
    dest_path: str = os.path.join(HERE, "data", "results", model_name, lc + ".tsv")
    df_write(results_df, dest_path)

    # Report results
    agg_result: AggregationRec = {
        "lc": lc,
        "num_sentences": results_df.shape[0],
        "inference_duration": results_df["item_inference_duration"].sum(),
        "total_duration": (datetime.now() - start_locale).total_seconds(),
        "avg_cer": results_df["cer"].mean(),
        "avg_wer": results_df["wer"].mean(),
        "avg_mer": results_df["mer"].mean(),
        "avg_wil": results_df["wil"].mean(),
        "avg_wip": results_df["wip"].mean(),
    }
    print(
        f"Finished LC={lc} for {agg_result['num_sentences']} sentences in {agg_result['total_duration']} secs. Avg CER={agg_result['avg_cer']} Avg WER={agg_result['avg_wer']}"
    )
    return agg_result


#
# MODEL HANDLER
#
def handle_model(model_name: str):
    print(f"==> Test run whisper model: {model_name}")
    # get the model
    model_dir: str = os.path.join(HERE, "data", "models", "default")
    # model: whisper.Whisper = whisper.load_model(name=model_name, download_root=model_dir, in_memory=True)
    # get a list of source test files
    diff_files: list[str] = glob.glob(os.path.join(HERE, "data", "cv-delta", "**", c.DIFF_FN), recursive=True)
    # create destination dir
    dest_path: str = os.path.join(HERE, "data", "results", model_name)
    os.makedirs(dest_path, exist_ok=True)

    # input records
    # inputs: list[HandleLocaleProps] = []
    # for p in diff_files:
    #     inputs.append({"model_name": model_name, "model": model, "diff_path": p})
    args = []
    for p in diff_files:
        args.append((model_name, p))

    # run them in parallel
    results: list[AggregationRec] = []
    
    # results.append(handle_locale(model_name, diff_files[48])) # Single test
    
    with mp.Pool(NUM_PROCS) as pool:
        results = pool.starmap(handle_locale, args)

    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.AGGREGATION_REC_COLS)
    df_write(results_df, os.path.join(HERE, "data", "results", model_name + "_summary.tsv"))

#
# MAIN PROCESS
#


def main() -> None:
    """Main process which loops through whisper models, test languages and handles locales with MP"""

    print(
        f"==> Forming a baseline using {len(conf.WHISPER_MODELS_TO_TEST)} whisper models: {conf.WHISPER_MODELS_TO_TEST}"
    )
    for model_name in conf.WHISPER_MODELS_TO_TEST:
        handle_model(model_name)


# Entry point
if __name__ == "__main__":
    main()
