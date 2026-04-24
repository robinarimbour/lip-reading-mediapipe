# Lip Reading using MediaPipe

This project implements a basic lip reading system using MediaPipe face mesh landmarks and an LSTM-based deep learning model.

> Early-stage project — accuracy and features are being improved.

---

## What it does

- Extracts lip landmarks from video using MediaPipe  
- Converts word segments into fixed-length sequences  
- Uses an LSTM-based model to learn temporal patterns in lip movements  
- Supports video and real-time (webcam) inference  

## Dataset

- GRID Corpus  
- 2 speakers used  
- 51 word classes  
- 9600 training samples  
- 2400 testing samples  

## Preprocessing

- Extract face mesh landmarks using MediaPipe  
- Select lip region landmarks  
- Normalize for translation, rotation, and scale  
- Convert word segments into sequences  
- Pad sequences to fixed length  

## Model

- LSTM-based sequence model for temporal learning  
- Takes sequences of lip landmarks as input  
- Outputs word-level predictions  

## Results

- Test Accuracy: **72.92%**

---

## Project Structure

    lip_reading/
    │
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
    │
    ├── src/                    # core logic
    │ ├── analysis/
    │ ├── preprocessing/
    │ ├── modeling/
    │ ├── utils/
    │ └── config.py
    │
    ├── scripts/                # runnable scripts
    │ └── analyze_data.py
    │ └── extract_landmarks.py
    │ └── predict_video.py
    │ └── realtime_demo.py
    │ └── visualize_landmarks.py
    │
    ├── models/                 # saved models
    │ └── v0/
    │
    └── notebooks/              # training
      └── train_model.ipynb


## Quick Start

1. Install dependencies  
```
pip install -r requirements.txt
```

2. Download the [GRID Corpus dataset](https://spandh.dcs.shef.ac.uk/avlombard/)

3. After downloading, extract the dataset and organize it as follows:
```
grid-corpus/
  └── data/
    ├── s1_processed/
    ├── s2_processed/
      ├── bbaf2n.mpg
      └── align/
        └── bbaf2n.align
```
4. Set dataset path `path/to/grid-corpus/data` in `.env`  

5. Run preprocessing  
```
py -m scripts.extract_landmarks
```

6. Train model  
```notebooks/train_model.ipynb```

7. Run inference  
```
py -m scripts.predict_video
```
```
py -m scripts.realtime_demo
```

---

## Limitations

- Limited to GRID dataset vocabulary  
- Struggles with visually similar words (e.g., "bin", "pin")  
- Performance depends on lighting and face alignment  
