import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

RUNS = [
    ("Zero-shot Validation", RESULTS / "zero_shot_validation_metrics.json"),
    ("Zero-shot Test", RESULTS / "zero_shot_test_metrics.json"),
    ("4-shot Validation", RESULTS / "four_shot_validation_metrics.json"),
    ("4-shot Test", RESULTS / "four_shot_test_metrics.json"),
    ("LoRA Validation", RESULTS / "lora_validation_metrics.json"),
    ("LoRA Test", RESULTS / "lora_test_metrics.json"),
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE = font(26, bold=True)
LABEL = font(18)
CELL = font(26, bold=True)
SMALL = font(15)


def load_matrix(path: Path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if "confusion_matrix" in metrics:
        return metrics["confusion_matrix"]
    return [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]


def text_center(draw, box, text, fnt, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=fnt, fill=fill)


def blue_for(value, max_value):
    if max_value <= 0:
        intensity = 0
    else:
        intensity = value / max_value
    start = (239, 246, 255)
    end = (30, 64, 175)
    return tuple(round(start[i] + (end[i] - start[i]) * intensity) for i in range(3))


def draw_matrix(title, matrix):
    width, height = 720, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((width / 2 - draw.textlength(title, font=TITLE) / 2, 24), title, font=TITLE, fill=(20, 43, 73))
    draw.text((282, 86), "Predicted label", font=LABEL, fill=(70, 70, 70))
    draw.text((30, 308), "True label", font=LABEL, fill=(70, 70, 70))

    labels = ["non_defective", "defective"]
    x0, y0 = 210, 150
    cell_w, cell_h = 210, 150
    max_value = max(max(row) for row in matrix)

    for j, label in enumerate(labels):
        text_center(draw, (x0 + j * cell_w, 116, x0 + (j + 1) * cell_w, 148), label, SMALL, (40, 40, 40))
    for i, label in enumerate(labels):
        text_center(draw, (30, y0 + i * cell_h, 195, y0 + (i + 1) * cell_h), label, SMALL, (40, 40, 40))

    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            box = (x0 + j * cell_w, y0 + i * cell_h, x0 + (j + 1) * cell_w, y0 + (i + 1) * cell_h)
            color = blue_for(value, max_value)
            draw.rectangle(box, fill=color, outline=(255, 255, 255), width=4)
            text_color = "white" if value > max_value * 0.55 else (20, 20, 20)
            text_center(draw, box, f"{value:,}", CELL, text_color)

    legend_y = 500
    draw.text((125, legend_y), "Matrix layout: [[TN, FP], [FN, TP]]", font=LABEL, fill=(70, 70, 70))
    draw.text((125, legend_y + 34), "Rows are true labels; columns are predicted labels.", font=SMALL, fill=(90, 90, 90))
    return img


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    images = []

    for title, path in RUNS:
        matrix = load_matrix(path)
        img = draw_matrix(title, matrix)
        output = FIGURES / f"{title.lower().replace(' ', '_')}_confusion_matrix.png"
        img.save(output)
        images.append((title, img))

    overview = Image.new("RGB", (1440, 1860), "white")
    draw = ImageDraw.Draw(overview)
    heading = "Confusion Matrices for Completed Runs"
    draw.text((720 - draw.textlength(heading, font=TITLE) / 2, 24), heading, font=TITLE, fill=(20, 43, 73))

    positions = [(0, 80), (720, 80), (0, 680), (720, 680), (0, 1280), (720, 1280)]
    for (_, img), pos in zip(images, positions):
        overview.paste(img, pos)

    overview_path = FIGURES / "confusion_matrices_overview.png"
    overview.save(overview_path)
    print(overview_path)


if __name__ == "__main__":
    main()
