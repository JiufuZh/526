NEGATIVE_LABEL_ID = 0
POSITIVE_LABEL_ID = 1
ID_TO_TEXT = {
    NEGATIVE_LABEL_ID: "non-defective",
    POSITIVE_LABEL_ID: "defective",
}
TEXT_TO_ID = {v: k for k, v in ID_TO_TEXT.items()}
LABEL_TEXTS = ["non-defective", "defective"]
