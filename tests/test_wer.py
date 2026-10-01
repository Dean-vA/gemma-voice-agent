from app.wer import normalize, word_errors


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Hello, robot! Can you hear me?") == ["hello", "robot", "can", "you", "hear", "me"]


def test_normalize_keeps_contractions_and_splits_hyphens():
    assert normalize("Don't use the well-known 'trick'.") == ["don't", "use", "the", "well", "known", "trick"]


def test_digits_match_spelled_out_numbers():
    assert normalize("75 degrees") == normalize("seventy-five degrees")
    assert normalize("at 3:45") == normalize("at three forty five")
    assert normalize("1,200 steps") == normalize("one thousand two hundred steps")
    assert normalize("2.5 metres, 40%") == normalize("two point five metres forty percent")


def test_accents_are_folded():
    assert normalize("Gabriel García Márquez") == ["gabriel", "garcia", "marquez"]


def test_word_errors_counts_substitution_deletion_insertion():
    assert word_errors("turn left at the door", "turn left at the door") == {"errors": 0, "ref_words": 5, "wer": 0.0}
    assert word_errors("turn left at the door", "turn right at door")["errors"] == 2
    assert word_errors("turn left", "please turn left now")["errors"] == 2


def test_empty_hypothesis_is_all_errors():
    assert word_errors("stop moving now", "") == {"errors": 3, "ref_words": 3, "wer": 1.0}
