

---

# ✅ README.md

# Sea Ice Motion Prediction (LSTM-Based)

This repository provides a lightweight implementation of an LSTM-based deep learning model for short-term Arctic sea‐ice motion prediction. The model uses optical flow fields derived from AMSR-E brightness temperature data as input and predicts future sea-ice drift.

## 📂 Dataset Sources

Below are the official data access portals used in this project:

* **AMSR-E L3 Sea Ice Data (NASA NSIDC)**
  [https://nsidc.org/data/AE_SI6](https://nsidc.org/data)

* **ECMWF ERA5 Reanalysis Meteorological Data**
  [https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)

* **IABP Arctic Buoy Observation Data**
  [https://iabp.apl.uw.edu/data.html](https://iabp.apl.uw.edu/data.html)

## 📁 Repository Structure

* `dataset.py` — Dataset loading and preprocessing
* `model.py` — LSTM-based prediction model
* `train.py` — Training script
* `main.py` — Inference and evaluation

## ⚙️ Requirements

Minimal dependencies:

* torch
* numpy
* xarray

Install via:

```
pip install -r requirements.txt
```

## 🚀 Quick Start

Train the model:

```
python train.py
```

Run inference:

```
python main.py
```

## 📜 License

This project is released under the MIT License.
See the `LICENSE` file for details.

---

# ✅ LICENSE

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

---



