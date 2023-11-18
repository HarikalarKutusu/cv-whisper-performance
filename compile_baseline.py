"""cv-tbox Whisper Performance - Evaluate against a test set (from default CV splits or custom splits) to form a baseline"""

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

# Standard lib
import os
import sys
import glob
import json
from datetime import datetime
import multiprocessing as mp
import logging
from typing import Dict, Any

# External dependencies
import pandas as pd
import psutil
import cvutils as cvu
import jiwer
import torch
import whisper

# Module
import config as conf
import const as c
from lib import AggregationRec, WhisperTranscriptionResult, TranscriptionRec
from lib import df_read, df_write, bytes2gb, dec2, dec6, lc_back_mapper

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

logging.getLogger("whisper").setLevel(logging.ERROR)  # get rid of warnings

# Multi Processing Cores
MAX_NUM_PROCS: int = psutil.cpu_count(logical=False)
MAX_NUM_PROCS = 4

# Common Voice Utilities Globals
cv: cvu.CV = cvu.CV()

#
# Whisper Models
#

model_dir: str = conf.WHISPER_MODELS_DIR
wmodel: whisper.Whisper
loaded_model: str = ""
device_mode: str = "cuda" if conf.USE_GPU else "cpu"


def load_model(requested_model: str) -> None:
    """Load model - if needed"""

    global model_dir
    global loaded_model
    global wmodel
    global device_mode

    if loaded_model != requested_model:
        loaded_model = requested_model
        wmodel = whisper.load_model(name=loaded_model, device=device_mode, download_root=model_dir)
        print("==> Model Loaded:", requested_model)


#
# Locale handling (multiprocessed)
#


def handle_locale(model_name: str, diff_path: str) -> AggregationRec:
    """Handle a single locale (multiprocess) against a whisper model"""

    global wmodel

    start_locale: datetime = datetime.now()
    # get dir
    locale_path: str = os.path.split(diff_path)[0]
    lc: str = locale_path.split(os.sep)[-1]
    dest_dir: str = os.path.join(HERE, c.EXPERIMENTS_DIR, conf.EXPERIMENT)
    dest_path: str = os.path.join(dest_dir, model_name, lc + ".tsv")
    trans_path: str = os.path.join(dest_dir, model_name, lc + ".json")
    # print("Processing:", lc)

    # model: whisper.Whisper = whisper.load_model(name=model_name, download_root=model_dir) # , in_memory=True
    load_model(model_name)  # , in_memory=True

    # get diff dataframe
    source_df: pd.DataFrame = df_read(diff_path)

    # Normalizer
    v = cvu.Validator(lc)

    # File for full trabscription results
    trans_file = open(trans_path, "w", encoding="utf8")
    trans_file.write("[\n")

    whisper_lc: str = lc_back_mapper(lc)

    # # sampling-related options
    # temperature: float = 0.0
    # sample_len: Optional[int] = None  # maximum number of tokens to sample
    # best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    # beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    # patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # # "alpha" in Google NMT, or None for length norm, when ranking generations
    # # to select which to return among the beams or best-of-N samples
    # length_penalty: Optional[float] = None

    # # text or tokens to feed as the prompt or the prefix; for more info:
    # # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    # prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    # prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # # list of tokens ids (or comma-separated token ids) to suppress
    # # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    # suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    # suppress_blank: bool = True  # this will suppress blank outputs

    # # timestamp sampling options
    # without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    # max_initial_timestamp: Optional[float] = 1.0

    options: Dict[str, Any] = {
        "task": "transcribe",
        "language": whisper_lc,
        "fp16": conf.USE_GPU,
        "beam_size": 5,
        "best_of": 5,
    }

    results: list[TranscriptionRec] = []
    # Loop through each record
    for inx, row in source_df.iterrows():
        start_row: datetime = datetime.now()
        result: TranscriptionRec = row.to_dict()  # type: ignore
        audio_path: str = os.path.join(locale_path, "clips", row["path"])
        start_transcription: datetime = datetime.now()
        transcription_result: WhisperTranscriptionResult = whisper.transcribe(
            model=wmodel, audio=audio_path, decode_options=options
        )  # type: ignore
        result["item_inference_duration"] = (datetime.now() - start_transcription).total_seconds()
        trans_file.write(json.dumps(transcription_result, ensure_ascii=False) + ",\n")  # save detailed response
        transcription_txt: str = transcription_result["text"].strip()
        is_ok, norm_transcription_txt = v.normalise(transcription_txt)
        result["transcription"] = transcription_txt
        result["norm_transcription"] = norm_transcription_txt
        # result["segments"] = transcription_result["segments"]
        result["detected_lc"] = transcription_result["language"]
        result["rtf"] = result["item_inference_duration"] / float(result["duration"])

        j_word: jiwer.WordOutput = jiwer.process_words(
            reference=row["norm_sentence"], hypothesis=norm_transcription_txt
        )
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
        "avg_rtf": dec6(results_df["rtf"].mean()),
    }
    print(
        f"Finished LC={lc} for {agg_result['num_sentences']} sentences in {agg_result['total_duration']} secs."
        + f" Avg CER={agg_result['avg_cer']} Avg WER={agg_result['avg_wer']}"
    )
    return agg_result


#
# MODEL HANDLER
#
def handle_model(model_name: str) -> None:
    """Processes a single model"""

    print(f"==> Test run whisper model: {model_name}")
    # get a list of source test files
    test_files: list[str] = glob.glob(
        os.path.join(HERE, c.TEST_SETS_DIR, conf.TEST_SET, "**", c.TEST_FN), recursive=True
    )
    test_files.sort()
    # create destination dir
    dest_path: str = os.path.join(HERE, c.EXPERIMENTS_DIR, conf.EXPERIMENT, model_name)
    os.makedirs(dest_path, exist_ok=True)

    # input records
    args = []
    for p in test_files:
        args.append((model_name, p))

    # run them in parallel
    results: list[AggregationRec] = []

    # results.append(handle_locale(model_name, diff_files[51])) # Single test

    # Decide on concurrency
    num_procs: int
    if conf.USE_GPU:
        vram_gb: float = bytes2gb(torch.cuda.mem_get_info(conf.GPU)[1])
        num_procs = min(int(vram_gb / c.WHISPER_MODEL_VRAM[model_name]), MAX_NUM_PROCS)
    else:  # whisper already uses concurrency on CPU, so not overload
        num_procs = min(2, psutil.cpu_count(logical=False))

    print(f"==> Using {num_procs} processes...")
    with mp.Pool(num_procs) as pool:
        results = pool.starmap(handle_locale, args)

    results_df: pd.DataFrame = pd.DataFrame.from_records(results, columns=c.AGGREGATION_REC_COLS)
    df_write(results_df, os.path.join(HERE, c.EXPERIMENTS_DIR, conf.EXPERIMENT, f"{model_name}_summary.tsv"))


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
