"""Word error rate for comparing transcription engines.

Pure Python (no dependencies) so scripts/asr_eval.py can import it on the host.
Text is normalized before scoring so engines aren't penalized for style:
case, punctuation and digits-vs-words ("75" == "seventy five") don't count.
"""
from __future__ import annotations

import re
import unicodedata

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _int_words(n % 100))
    if n < 1_000_000:
        return _int_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _int_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


def _number_words(match: re.Match) -> str:
    token = match.group(0).replace(",", "")
    if ":" in token:  # clock time: 3:45 -> three forty five, 7:05 -> seven oh five, 9:00 -> nine
        hours, minutes = token.split(":")
        m = int(minutes)
        tail = "" if m == 0 else (" oh " + _ONES[m] if m < 10 else " " + _int_words(m))
        return " " + _int_words(int(hours)) + tail + " "
    if "." in token:
        whole, frac = token.split(".")
        return " " + _int_words(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in frac) + " "
    return " " + _int_words(int(token)) + " "


def normalize(text: str) -> list[str]:
    """Lowercase words with punctuation removed and numbers spelled out."""
    text = text.lower().replace("’", "'")
    # Fold accents so "García" and "Garcia" count as the same word.
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.replace("%", " percent ").replace("&", " and ")
    text = re.sub(r"\d{1,2}:\d{2}|\d+(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+", _number_words, text)
    text = re.sub(r"[^a-z' ]+", " ", text)          # drop punctuation, incl. hyphens
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # keep apostrophes only inside words
    return text.split()


def word_errors(reference: str, hypothesis: str) -> dict:
    """Edit distance between the normalized word sequences.

    Returns ``{errors, ref_words, wer}``; aggregate WER over many clips is
    sum(errors) / sum(ref_words), not the mean of per-clip WERs.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    errors = prev[-1]
    return {"errors": errors, "ref_words": len(ref), "wer": round(errors / max(1, len(ref)), 4)}
