from defect_detection.label_norm import normalize_label


def test_non_defective_checked_before_defective():
    assert normalize_label("non-defective") == 0
    assert normalize_label("The answer is non defective.") == 0


def test_defective_variants():
    assert normalize_label("defective") == 1
    assert normalize_label("buggy") == 1
