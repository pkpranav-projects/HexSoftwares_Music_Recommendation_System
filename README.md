<div align="center">

# SonicMatch: Content-Based Music Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NearestNeighbors](https://img.shields.io/badge/Algorithm-KNN%20Cosine%20Similarity-blueviolet?style=for-the-badge)](https://scikit-learn.org/stable/modules/neighbors.html)
[![Spotify Dataset](https://img.shields.io/badge/Data-Spotify%20Audio%20Features-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=for-the-badge)](https://github.com/pkpranav-projects/HexSoftwares_Music_Recommendation_System/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An intelligent content-based audio recommendation system that maps songs into a multi-dimensional acoustic vector space, computing high-dimensional cosine proximity to deliver personalized music discoveries without requiring user history.

Developed as part of the **Hex Softwares Machine Learning Internship**.

[Core Concept](#-core-concept--mathematical-engine) • [Audio Feature Matrix](#-acoustic-feature-matrix) • [Architecture](#-architecture--pipeline) • [Quickstart](#-quickstart--installation) • [Interactive Usage](#-interactive-usage--sample-output) • [Author](#-author)

</div>

---

## Overview

Unlike collaborative filtering systems that suffer from the "cold-start" problem when a user has no listening history, **SonicMatch** leverages **Content-Based Acoustic Vector Filtering**. 

By analyzing the intrinsic auditory characteristics of tracks—ranging from rhythmic speed and perceived loudness to melodic valence and danceability—the engine locates the nearest harmonic neighbors in 5D feature space using **K-Nearest Neighbors (KNN)** with a **Cosine Distance Metric**.

---

## Core Concept & Mathematical Engine

Each audio track in the dataset is represented as a 5-dimensional feature vector:

$$\mathbf{x} = [\text{danceability}, \text{energy}, \text{tempo}, \text{loudness}, \text{valence}]^T$$

### 1. Standardization (Z-Score)
To prevent variables with larger numerical ranges (such as `tempo` in BPM $\approx 60-200$ and `loudness` in dB $\approx -60 \text{ to } 0$) from dominating normalized metrics ($0.0 \text{ to } 1.0$), all features undergo standard scaling:

$$z = \frac{x - \mu}{\sigma}$$

### 2. Cosine Distance Metric
Given two standardized song vectors $\mathbf{u}$ and $\mathbf{v}$, the cosine similarity represents the normalized dot product (the cosine of the angle between them):

$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

The cosine distance is given by:

$$D_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

The $K$-Nearest Neighbors model indexes the dataset with $K = 6$ (query track + top 5 nearest sonic counterparts).

---

## Acoustic Feature Matrix

| Feature | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **`danceability`** | Float | `0.0 – 1.0` | How suitable a track is for dancing based on tempo, rhythm stability, beat strength, and regularity. |
| **`energy`** | Float | `0.0 – 1.0` | Perceptual measure of intensity and activity. Energetic tracks feel fast, loud, and noisy. |
| **`tempo`** | Float | `BPM` | The overall estimated tempo of a track in beats per minute (BPM). |
| **`loudness`** | Float | `dB (-60 to 0)` | Overall loudness of a track in decibels (dB), averaged across the entire composition. |
| **`valence`** | Float | `0.0 – 1.0` | Musical positiveness. High valence sounds happy, cheerful, euphoric; low valence sounds sad, depressed, angry. |

---

## Architecture & Pipeline

```text
  +--------------------------------+
  |      SpotifyFeatures.csv       |  <-- Thousands of catalogued audio tracks
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  |  Stratified Feature Extraction |  <-- 5 Acoustic Dimensions
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  |    StandardScaler Fitting      |  <-- Zero-mean, unit-variance transform
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  | NearestNeighbors(metric='cos') |  <-- High-dimensional spatial index
  +--------------------------------+
                  |
                  +---------------------------+
                  |                           |
                  v                           v
         [ User Query: Song ]        [ 5 Nearest Neighbors ]
                  |                           |
                  +------------> <------------+
                                |
                                v
               Top 5 Curated Track Recommendations
```

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pkpranav-projects/HexSoftwares_Music_Recommendation_System.git
cd HexSoftwares_Music_Recommendation_System
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(If requirements.txt is minimal, ensure the core scientific stack is present)*:
```bash
pip install pandas scikit-learn numpy
```

### 4. Unpack Dataset (If Archive Present)
```bash
# If SpotifyFeatures.csv is compressed:
python -c "import zipfile; zipfile.ZipFile('SpotifyFeatures.zip').extractall('.') if not __import__('os').path.exists('SpotifyFeatures.csv') else None"
```

---

## Interactive Usage & Sample Output

Launch the recommendation script:

```bash
python music_recommendation.py
```

**Interactive Terminal Session:**
```text
Enter a song name: Shape of You

Recommended Songs:
1. Despacito - Remix
2. Attention
3. Cheap Thrills
4. Rockabye (feat. Sean Paul & Anne-Marie)
5. Closer
```

*(If an uncatalogued song is entered)*:
```text
Enter a song name: Unknown Indie Track
Song not found in dataset
```

---

## Repository Structure

```text
HexSoftwares_Music_Recommendation_System/
├── SpotifyFeatures.zip     # Compressed Spotify audio catalog
├── music_recommendation.py # Core KNN acoustic recommendation engine
├── requirements.txt        # Python dependency manifest
├── data/                   # Data staging directory
├── src/                    # Modular source extensions
├── .gitignore              # Environment exclusions
├── LICENSE                 # MIT License
└── README.md               # Architecture & usage documentation
```

---

## Internship Details

- **Organization**: Hex Softwares
- **Domain**: Machine Learning / Recommender Systems
- **Project**: Content-Based Music Recommendation Engine
- **Intern**: **P K PRANAV** ([@pkpranav-projects](https://github.com/pkpranav-projects))

## License

This project is open-source under the [MIT License](LICENSE).