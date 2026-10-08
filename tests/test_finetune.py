from scripts.finetune_whisper import normalize_label


def test_labels_are_rewritten_in_model_style() -> None:
    raw = "Mahallani oqsoqoli,\nkim qanaqa — kambag'almi, yo'qmi? Ob-havo; ma'no"
    assert normalize_label(raw) == "mahallani oqsoqoli kim qanaqa kambagʻalmi yoʻqmi ob havo maʼno"


def test_existing_model_style_is_kept() -> None:
    assert normalize_label("toʻgʻri maʼno") == "toʻgʻri maʼno"
