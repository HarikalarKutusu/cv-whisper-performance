"""cv-tbox Whisper Performance - Libraries"""
###########################################################################
# lib.py
#
# Library for this module
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard lib
import os
import sys
import csv
from itertools import takewhile, repeat

# External dependencies
import pandas as pd

# Module
import const as c
import conf


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
    """Writes out a dataframe to a file."""
    # Create/override the file
    df.to_csv(fpath, header=True, index=False, encoding="utf-8", sep="\t", escapechar="\\", quoting=csv.QUOTE_NONE)
    if conf.VERBOSE:
        print(f"Generated: {fpath} Records={df.shape[0]}")
    return True


def line_count(p: str) -> int:
    """
    Efficiently count the number of lines in a file

    Arguments:
    p: path to count the number of lines in

    Returns: int value (number of lines in file)
    """
    with open(p, "rb") as fd:
        bufgen = takewhile(lambda x: x, (fd.raw.read(1024 * 1024) for _ in repeat(None)))
    res: int = sum(buf.count(b"\n") if buf else 0 for buf in bufgen)
    return res


def lc_mapper(lc: str) -> str:
    """Map from Whisper language to CV lc code"""
    if lc in c.LC_MAPPER.keys():
        return c.LC_MAPPER[lc]
    else:
        return lc


def lc_back_mapper(lc: str) -> str:
    """Map from CV lc code to Whisper language"""
    if lc in c.LC_BACK_MAPPER.keys():
        return c.LC_BACK_MAPPER[lc]
    else:
        return lc


def dec2(x: float) -> float:
    """Return rounded float to two decimals"""
    return round(x, 2)


def dec6(x: float) -> float:
    """Return rounded float to six decimals"""
    return round(x, 6)


def bytes2gb(mem: int) -> float:
    """Convert bytes to gigabytes with two decimals"""
    return dec2(mem / (1024 * 1024 * 1024))
