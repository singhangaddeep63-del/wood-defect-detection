# Wood Defect Detection

Deep learning model that classifies wood boards as good or defective, with a Gradio demo app.

## Demo

![Demo of the wood defect detection app](assets/demo.png)

The app takes a photo of a wood board and returns "good" or "defect" with a score (threshold 0.4).

## Error analysis (20 sample boards: 18 correct)

- defect_5.jpg (true: defect, predicted: good, score 0.366). The board shows only small, low-contrast marks (a tiny brown spot, a thin streak, and small surface bumps), which are easy to lose when the wide image is resized to a square input.
- good_2.jpg (true: good, predicted: defect, score 0.637). The board contains a visible knot, which the model likely associates with defects.

Next steps: crop the black borders, test a higher input resolution, and review how the dataset labels small knots.

## Dataset

The model was trained on the public wood surface defects dataset by Kodytek, Bodzas and Bilik, hosted on Kaggle (about 20.3k files, 4.42 GB, CC0 public domain license).

Citation: Kodytek P, Bodzas A, Bilik P. A large-scale image dataset of wood surface defects for automated vision-based quality control processes. F1000Research 2021, 10:581. https://doi.org/10.12688/f1000research.52903.1