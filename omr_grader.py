"""
OMR Answer Sheet Grader
-----------------------
Grades scanned 4-option answer sheets (165 questions) against an answer-key
screenshot and exports per-question status and per-sheet scores to CSV.

Usage (from the project folder):
    python omr_grader.py --sheets sample/sheets --key sample/answer_key.png --out output

Input layout
    sample/sheets/*.tif   scanned student answer sheets (same printed layout)
    sample/answer_key.png result/key screenshot (green marker = correct option)

Output (written to --out)
    save_all_status.csv   Q_num + one column per sheet: True / False / "-" (blank)
    save_all_score.csv    file_name, status_student (percentage score)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# ---------------------------------------------------------------- constants
N_QUESTIONS = 165
DELETED_QUESTIONS = {149}               # question numbers excluded from scoring
DELETED_LABEL = "this question has been deleted"
BLANK_LABEL = "-"

# Sheet: every sheet is first resized to this size, then 17 blocks are cropped
# as (y1, y2, x1, x2). Each block holds 10 questions x 4 options; the last 5
# questions of the 17th block do not exist (170 - 5 = 165).
SHEET_SIZE = (1400, 1560)               # (width, height)
SHEET_BLOCKS = [
    (409, 600, 110, 236), (615, 810, 115, 236), (825, 1014, 115, 236), (1029, 1214, 118, 236), (1229, 1416, 118, 236),
    (414, 600, 330, 456), (615, 804, 332, 456), (825, 1004, 337, 456), (1029, 1214, 337, 456), (1229, 1416, 337, 456),
    (409, 600, 553, 676), (615, 810, 553, 676), (825, 1014, 555, 676), (1029, 1214, 555, 676), (1234, 1416, 558, 676),
    (409, 600, 772, 896), (615, 810, 775, 896),
]
BLOCK_SIZE = (200, 250)                 # each cropped block is resized to this
FILLED_MIN_WHITE = 900                  # white pixels needed to call a bubble filled

# Answer key: block tops below are measured on the original ~16000 px tall
# screenshot; they are scaled by (image height / KEY_REF_HEIGHT) so smaller or
# larger exports of the same screenshot also work.
KEY_REF_HEIGHT = 16000
KEY_BLOCK_TOPS = [410, 1350, 2300, 3230, 4160, 5100, 6050, 7000, 7910,
                  8850, 9800, 10740, 11660, 12600, 13550, 14500, 15430]
KEY_FIRST_ROW_OFFSET = 56               # block top -> centre of its first row
KEY_ROW_PITCH = 87.6                    # distance between question rows
KEY_COLUMN_EDGES = (0.287, 0.515, 0.742)  # fractions of image width -> options 1..4
KEY_MIN_BLOB_AREA = 8                   # smallest green marker (pixels)


# ---------------------------------------------------------- reading a sheet
def read_sheet(path) -> list[int]:
    """Return 165 values: 1-4 = filled option, 0 = blank, -1 = 3+ bubbles filled."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, SHEET_SIZE)
    answers: list[int] = []
    for y1, y2, x1, x2 in SHEET_BLOCKS:
        block = cv2.resize(img[y1:y2, x1:x2], BLOCK_SIZE)
        _, block = cv2.threshold(block, 127, 255, cv2.THRESH_BINARY_INV)
        for row in np.vsplit(block, 10):
            white = [int(np.sum(cell == 255)) for cell in np.hsplit(row, 4)]
            filled = sum(w > FILLED_MIN_WHITE for w in white)
            if filled == 0:
                answers.append(0)
            elif filled <= 2:           # 2 filled -> darkest one wins (as in the original)
                answers.append(int(np.argmax(white)) + 1)
            else:
                answers.append(-1)
    return answers[:N_QUESTIONS]


# ------------------------------------------------------- reading the key
def read_key(path) -> list[int]:
    """Return 165 values: correct option 1-4, or 0 where no marker was found."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    height, width = img.shape[:2]
    scale = height / KEY_REF_HEIGHT

    # Green markers = correct answer (badge or check mark). Red markers
    # (the student's wrong choice) are ignored.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85)
             & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 100))
    green = ndi.binary_closing(green, iterations=2)
    labels, _ = ndi.label(green)
    centres = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if (labels[sl] == i).sum() < KEY_MIN_BLOB_AREA:
            continue
        centres.append(((sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2))

    pitch = KEY_ROW_PITCH * scale              # distance between question rows (px)
    tol = 14 * scale                           # max distance of a marker from its row centre
    edges = [e * width for e in KEY_COLUMN_EDGES]
    key = [0] * N_QUESTIONS
    for b, top in enumerate(KEY_BLOCK_TOPS):
        rows = min(10, N_QUESTIONS - b * 10)
        first0 = (top + KEY_FIRST_ROW_OFFSET) * scale
        # Layout drifts a little from block to block (ads, spacing): search the
        # vertical shift that makes the most markers sit on this block's row grid.
        best_shift, best_score = 0.0, -1
        for shift in np.arange(-140 * scale, 140 * scale + 0.5, 0.5):
            f, score = first0 + shift, 0
            for cy, _ in centres:
                r = round((cy - f) / pitch)
                if 0 <= r < rows and abs(cy - (f + r * pitch)) < tol:
                    score += 1
            if score > best_score or (score == best_score and abs(shift) < abs(best_shift)):
                best_shift, best_score = shift, score
        f = first0 + best_shift
        for cy, cx in centres:
            r = round((cy - f) / pitch)
            if 0 <= r < rows and abs(cy - (f + r * pitch)) < tol:
                key[b * 10 + r] = int(np.digitize(cx, edges)) + 1
    return key


# ------------------------------------------------------------- grading
def grade(answers: list[int], key: list[int]) -> list:
    """Per-question status: True / False / '-' (blank) / DELETED_LABEL."""
    status = []
    for i, (a, k) in enumerate(zip(answers, key), start=1):
        if i in DELETED_QUESTIONS:
            status.append(DELETED_LABEL)
        elif a == 0:
            status.append(BLANK_LABEL)
        else:
            status.append(bool(a == k and k != 0))
    return status


def score_percent(status: list) -> float:
    graded = N_QUESTIONS - len(DELETED_QUESTIONS)
    return round(sum(s is True for s in status) / graded * 100, 2)


def detail_table(answers: list[int], key: list[int]) -> pd.DataFrame:
    """Readable table (op1..op4, status, key) for one sheet, like the original notebook."""
    status = grade(answers, key)
    data = {"Q": range(1, N_QUESTIONS + 1)}
    for opt in range(1, 5):
        data[f"op{opt}"] = [str(opt) * 4 if a == opt else "...." for a in answers]
    data["status_student"] = ["X" if s == BLANK_LABEL else s for s in status]
    data["key"] = key
    return pd.DataFrame(data).set_index("Q")


def grade_folder(sheets_dir, key_path, out_dir="output") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grade every .tif/.tiff in sheets_dir and write the two CSV files."""
    sheets = sorted(p for p in Path(sheets_dir).iterdir()
                    if p.suffix.lower() in (".tif", ".tiff"))
    if not sheets:
        raise FileNotFoundError(f"No .tif sheets found in {sheets_dir}")
    key = read_key(key_path)
    missing = [i + 1 for i, k in enumerate(key)
               if k == 0 and (i + 1) not in DELETED_QUESTIONS]
    if missing:
        warnings.warn(f"No key marker found for questions {missing}; "
                      "they are graded as incorrect.")
    status_cols, scores = {}, []
    for p in sheets:
        status = grade(read_sheet(p), key)
        status_cols[p.stem] = status
        scores.append((p.stem, score_percent(status)))
    status_df = pd.DataFrame(status_cols)
    status_df.insert(0, "Q_num", range(1, N_QUESTIONS + 1))
    score_df = pd.DataFrame(scores, columns=["file_name", "status_student"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(out / "save_all_status.csv", index=False)
    score_df.to_csv(out / "save_all_score.csv", index=False)
    return status_df, score_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade scanned OMR answer sheets.")
    ap.add_argument("--sheets", default="sample/sheets", help="folder with .tif sheets")
    ap.add_argument("--key", default="sample/answer_key.png", help="answer-key image")
    ap.add_argument("--out", default="output", help="folder for the CSV results")
    args = ap.parse_args()
    _, scores = grade_folder(args.sheets, args.key, args.out)
    print(scores.to_string(index=False))
    print(f"\nResults written to: {args.out}/")


if __name__ == "__main__":
    main()
