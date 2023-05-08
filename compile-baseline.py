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

import sys, os, shutil, glob, csv, json
from datetime import datetime, timedelta
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

# whisper
import whisper

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
    dec2,
    dec6,
    df_read,
    df_write,
    CommonVoiceRec,
    TranscriptionRec,
)

# Multi Processing Cores
MAX_NUM_PROCS: int = psutil.cpu_count(logical=False)

# Common Voice Utilities Globals
cv: cvu.CV = cvu.CV()

#
# Whisper Models
#

model_dir: str = os.path.join(HERE, "data", "models", "default")
WModel: whisper.Whisper
LoadedModel: str = ""
DeviceMode: str = "cuda" if conf.USE_GPU else "cpu"

def loadModel(requestedModel: str):
    global model_dir
    global LoadedModel
    global WModel
    global DeviceMode

    device: str = "cuda" if conf.USE_GPU else "cpu"

    if LoadedModel != requestedModel:
        LoadedModel = requestedModel
        WModel = whisper.load_model(name=LoadedModel, device=DeviceMode, download_root=model_dir)
        print("==> Model Loaded:", requestedModel)


#
# Locale handling (multiprocessed)
#

def handle_locale(model_name: str, diff_path: str) -> AggregationRec:
    """Handle a single locale (multiprocess)"""
    global WModel

    start_locale: datetime = datetime.now()
    # get dir
    locale_path: str = os.path.split(diff_path)[0]
    lc: str = locale_path.split(os.sep)[-1]
    dest_path: str = os.path.join(HERE, "data", "results", model_name, lc + ".tsv")
    trans_path: str = os.path.join(HERE, "data", "results", model_name, lc + ".txt")
    # print("Processing:", lc)

    # model: whisper.Whisper = whisper.load_model(name=model_name, download_root=model_dir) # , in_memory=True
    loadModel(model_name) # , in_memory=True

    # get diff dataframe
    source_df: pd.DataFrame = df_read(diff_path)

    # Normalizer
    v = cvu.Validator(lc)

    # File for full trabscription results
    trans_file =  open(trans_path, "w", encoding="utf8")
    trans_file.write("[\n")

    results: list[TranscriptionRec] = []
    # Loop through each record
    for inx, row in source_df.iterrows():
        start_row: datetime = datetime.now()
        result: TranscriptionRec = row.to_dict()  # type: ignore
        audio_path: str = os.path.join(locale_path, "clips", row["path"])
        start_transcription: datetime = datetime.now()
        transcription_result: WhisperTranscriptionResult = whisper.transcribe(model=WModel, audio=audio_path)  # type: ignore
        result["item_inference_duration"] = (datetime.now() - start_transcription).total_seconds()
        trans_file.write(json.dumps(transcription_result, ensure_ascii=False) + "\n") # save detailed response
        transcription_txt: str = transcription_result["text"].strip()
        isOK, norm_transcription_txt = v.normalise(transcription_txt)
        result["transcription"] = transcription_txt
        result["norm_transcription"] = norm_transcription_txt
        # result["segments"] = transcription_result["segments"]
        result["detected_lc"] = transcription_result["language"]
        result["rtf"] = result["item_inference_duration"] / float(result["duration"])

        j_word: jiwer.WordOutput = jiwer.process_words(reference=row["norm_sentence"], hypothesis=norm_transcription_txt)
        j_char: jiwer.CharacterOutput = jiwer.process_characters(
            reference=row["norm_sentence"], hypothesis=norm_transcription_txt
        )
        result["cer"] = j_char.cer
        result["wer"] = j_word.wer
        result["mer"] = j_word.mer
        result["wil"] = j_word.wil
        result["wip"] = j_word.wip
        result["item_total_duration"] = (datetime.now() - start_row).total_seconds()
        results.append(result)  # type: ignore

    # save locale results for this model
    trans_file.write("]\n")
    trans_file.close()
    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.TRANSCRIPTION_REC_COLS)
    df_write(results_df, dest_path)

    # Report results
    agg_result: AggregationRec = {
        "model": model_name,
        "lc": lc,
        "num_sentences": results_df.shape[0],
        "duration": dec2(results_df["duration"].sum()),
        "avg_char_speed": dec2(results_df["char_speed"].mean()),
        "inference_duration": dec2(results_df["item_inference_duration"].sum()),
        "total_duration": dec2((datetime.now() - start_locale).total_seconds()),
        "avg_cer": dec6(results_df["cer"].mean()),
        "avg_wer": dec6(results_df["wer"].mean()),
        "avg_mer": dec6(results_df["mer"].mean()),
        "avg_wil": dec6(results_df["wil"].mean()),
        "avg_wip": dec6(results_df["wip"].mean()),
        "avg_rtf": dec6(results_df["rtf"].mean())
    }
    print(
        f"Finished LC={lc} for {agg_result['num_sentences']} sentences in {agg_result['total_duration']} secs. Avg CER={agg_result['avg_cer']} Avg WER={agg_result['avg_wer']}"
    )
    return agg_result


#
# MODEL HANDLER
#
def handle_model(model_name: str) -> None:
    print(f"==> Test run whisper model: {model_name}")
    # get a list of source test files
    diff_files: list[str] = glob.glob(os.path.join(HERE, "data", "cv-delta", "**", c.DIFF_FN), recursive=True)
    diff_files.sort()
    # create destination dir
    dest_path: str = os.path.join(HERE, "data", "results", model_name)
    os.makedirs(dest_path, exist_ok=True)

    # input records
    args = []
    for p in diff_files:
        args.append((model_name, p))

    # run them in parallel
    results: list[AggregationRec] = []
    
    # results.append(handle_locale(model_name, diff_files[51])) # Single test

    # Decide on concurrency
    NUM_PROCS: int = min(
        int(conf.VRAM / c.WHISPER_MODEL_VRAM[model_name]),
        MAX_NUM_PROCS
        )

    print(f"==> Using {NUM_PROCS} processes...")
    with mp.Pool(NUM_PROCS) as pool:
        results = pool.starmap(handle_locale, args)

    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.AGGREGATION_REC_COLS)
    df_write(results_df, os.path.join(HERE, "data", "results", f"{model_name}_summary.tsv"))

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
    # combine results


# Entry point
if __name__ == "__main__":
    main()
