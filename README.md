# OMR Answer Sheet Grader

Automatic grading of scanned multiple-choice answer sheets using classical computer vision (OpenCV).
University project 

## Overview
The program reads scanned answer sheets (4 options per question, 165 questions), detects the filled
bubble for each question, compares the answers with an answer key, and exports a per-question status
table and a percentage score per sheet.

## How it works
1. **Sheet:** the scan is resized to a fixed size and cropped into 17 blocks of 10 questions. Each block is
   thresholded, split into rows and 4 option columns, and the filled option is the one with the most white
   pixels (a bubble counts as filled above 900 white pixels).
2. **Key:** the answer-key screenshot is read by colour: green markers (badge or check mark) are the correct
   options; red markers (the student's wrong choice) are ignored. Rows are aligned automatically per block,
   so the key image can have a different resolution than the one the coordinates were measured on.
3. **Grading:** score = correct answers / 164 × 100 (question 149 is deleted and excluded; no negative marking).

## Project structure
```
omr-answer-sheet-grader/
├── README.md
├── requirements.txt
├── .gitignore
├── omr_grader.py             # main code (functions + command-line tool)
├── omr_grader.ipynb          # notebook that runs the pipeline on sample/
├── sample/
│   ├── answer_key.png        # answer-key screenshot
│   └── sheets/
│       └── image0000050A.tif # one scanned answer sheet
└── results/
    ├── save_all_score.csv    # scores of the original 23 sheets
    └── save_all_status.csv   # per-question status of the original 23 sheets
```

## Setup
```bash
git clone https://github.com/<your-username>/omr-answer-sheet-grader.git
cd omr-answer-sheet-grader
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python omr_grader.py --sheets sample/sheets --key sample/answer_key.png --out output
```
Put any number of `.tif` scans (same printed sheet layout) in `sample/sheets/`.
Output: `output/save_all_status.csv` and `output/save_all_score.csv`.
Or open `omr_grader.ipynb` and run all cells.

## Results
On the included sample sheet (`image0000050A.tif`) the program reproduces the saved result exactly:
165 of 165 question statuses identical, 41 correct answers, score 25.0 %.
`results/` holds the outputs of the original run on 23 sheets (range 4.88 – 43.90 %).
Only this one sheet was re-verified; the other 22 scans are not in the repository.
Accuracy against manual grading has not been measured.

## Limitations
- Block coordinates are fixed for one specific sheet layout.
- The key reader expects the same screenshot layout (4 option columns, green markers).
- A question with 3 or more filled bubbles is read as invalid (-1) and graded incorrect.
- Question 149 is hard-coded as deleted (`DELETED_QUESTIONS` in `omr_grader.py`).
- No automated tests.

## Data notice
The sample scan belongs to a student and the key screenshot comes from a third-party exam website.
Confirm you may redistribute them (or remove / anonymise them) before making the repository public.

## Future improvements
- Detect the answer grid automatically instead of using fixed coordinates.
- Add tests and measure accuracy against manual grading.

## Author
Hasti — Computer Engineering student. 