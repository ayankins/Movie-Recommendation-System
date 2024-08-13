# Movies Recommendation System

This repository contains a machine learning model for recommending movies based on user preferences. The model utilizes textual features from movie metadata to provide similar movie suggestions.

## Features

- **Data Collection and Preprocessing**:
  - Loads movie data from a CSV file into a Pandas DataFrame.
  - Handles missing values and combines relevant features such as genres, keywords, tagline, cast, and director.

- **Recommendation Engine**:
  - Uses TF-IDF Vectorization to convert text data into feature vectors.
  - Computes cosine similarity to find similar movies based on user input.

- **User Interaction**:
  - Allows users to input their favorite movie and provides a list of recommended similar movies.

## Libraries Used

```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
