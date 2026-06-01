from defect_detection.prompts import build_prompt


def test_prompt_contains_required_label_constraint():
    p = build_prompt("int f(){return 0;}")
    assert "defective" in p
    assert "non-defective" in p
    assert "### Answer" in p
