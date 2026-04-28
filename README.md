# Lip Reading using MediaPipe

This project implements a basic lip reading system using MediaPipe face mesh landmarks and an LSTM-based deep learning model.

![Lip Reading Demo](assets/demo.png)

---

## What it does

- Extracts lip landmarks from video using MediaPipe  
- Applies geometric normalization (translation, rotation, scaling)  
- Converts word segments into fixed-length sequences  
- Trains an LSTM-based model to learn temporal lip movement patterns  
- Supports:
  - Video inference  
  - GRID sentence prediction with alignment  
  - Real-time webcam inference  
  - Streamlit web app interface  

## Dataset

- GRID Corpus  
- Vocabulary: 51 words  
- Model was trained on:
  - 32 speakers
  - 130 clips per speaker

## Preprocessing

- Extract face mesh landmarks using MediaPipe  
- Select lip landmarks only  
- Normalize per frame:
  - Centering (translation)
  - Rotation correction (tilt)
  - Scale normalization (face width)
- Segment videos using alignment files  
- Convert word segments into sequences  
- Pad sequences to fixed length 

## Model

- LSTM-based sequence model  
- Input: sequence of lip landmark coordinates  
- Output: word-level classification

## Results

- Train Accuracy: ~**69%**
- Validation Accuracy: ~**60%**

---

## Project Structure

    lip_reading/
    │
    ├── .env
    ├── .gitignore
    ├── requirements.txt
    ├── README.txt
    │
    ├── src/
    │   ├── analysis/
    │   │   ├── dataset_analysis.py
    │   ├── modeling/
    │   │   ├── model_loader.py
    │   │   ├── predictor.py
    │   │   └── realtime.py
    │   ├── preprocessing/
    │   │   ├── align.py
    │   │   ├── landmarks.py
    │   │   ├── pipeline.py
    │   │   └── video.py
    │   ├── utils/
    │   │   ├── dataset.py
    │   │   └── visualization.py
    │   └── config.py
    │
    ├── scripts/
    │   ├── analyze_data.py
    │   ├── extract_landmarks.py
    │   ├── predict_video.py
    │   ├── realtime_demo.py
    │   ├── save_clips.py
    │   └── visualize_landmarks.py
    │
    ├── models/
    │   └── v1/
    │
    ├── notebooks/
    │   └── train_model.ipynb
    │
    └── app.py   (Streamlit interface)


## Quick Start

1. Install dependencies  
`pip install -r requirements.txt`

2. Download the [GRID Corpus dataset](https://spandh.dcs.shef.ac.uk/gridcorpus/)

3. After downloading, extract the dataset and organize it as follows:
```
grid-corpus/
  └── data/
      ├── s1/
      ├── s2/
      │   ├── video.mpg
      │   └── align/
      │       └── video.align
```

4. Extract landmarks

All speakers:  
```
py -m scripts.extract_landmarks --data_path path/to/grid
```

Specific speakers:  
```
py -m scripts.extract_landmarks --data_path path/to/grid --speakers s1 s2
```

Range:  
```
py -m scripts.extract_landmarks --data_path path/to/grid --speakers s1-s20
```

Optional arguments:  
`--split train`  
`--num_samples 100`  
`--output_dir path/to/output`

5. Train model

Use:  
`notebooks/train_model.ipynb`

6. Run inference

GRID video:
```
py -m scripts.predict_video --model_dir models/v1 --mode grid --video_path path/to/video.mpg --align_path path/to/file.align
```

Realtime webcam:
```
py -m scripts.realtime_demo --model_dir models/v1
```

7. Run Streamlit app

```
streamlit run app.py
```

---

## Limitations

- Performance drops significantly on unseen speakers
- Limited vocabulary (GRID dataset constraint)
- Sensitive to lighting and face detection quality
