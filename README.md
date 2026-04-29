# Supervised Learning

This repository contains a simple supervised learning example written in Python.
It implements a k-nearest neighbors classifier from scratch, trains it on labeled
student-performance data, and predicts whether new students are likely to pass or
fail.

## Files

- `supervised_learning.py` - pure-Python k-nearest neighbors classifier and demo.
- `.gitignore` - common Python files and local environment folders to ignore.

## Run

```bash
python supervised_learning.py
```

Example output:

```text
Student with features [2.5, 55.0] is predicted to: fail
Student with features [6.5, 88.0] is predicted to: pass
```

## How It Works

Supervised learning trains a model with examples that already have labels. In
this project, each training row has input features such as study hours and
attendance percentage, plus a label: `pass` or `fail`.

The k-nearest neighbors algorithm predicts a new label by:

1. Measuring the distance between the new sample and every training sample.
2. Selecting the `k` closest training samples.
3. Returning the most common label among those neighbors.
