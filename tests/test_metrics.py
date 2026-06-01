from defect_detection.metrics import compute_binary_metrics


def test_metrics_basic():
    m = compute_binary_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert "macro_f1" in m
    assert m["tn"] == 2
    assert m["tp"] == 1
