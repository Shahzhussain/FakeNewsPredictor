# Fake News Detector

## Overview
The Fake News Detector is a Python-based project designed to preprocess and analyze news datasets to identify fake news. It leverages machine learning techniques to classify news articles as real or fake.

## Project Structure
```
FakeNewsDetector/
├── app.py                # Main application script
├── preprocess.py         # Preprocessing script for data cleaning
├── preprocess2.py        # Additional preprocessing utilities
├── requirements.txt      # Python dependencies
├── dataset/              # Folder containing datasets
│   ├── news.csv          # Raw news dataset
│   └── news_preprocessed.csv # Preprocessed news dataset
```

## Dataset
- `news.csv`: The raw dataset containing news articles.
- `news_preprocessed.csv`: The cleaned and preprocessed dataset ready for analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FakeNewsDetector
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess the dataset:
   ```bash
   python preprocess.py
   ```
2. Run the main application:
   ```bash
   python app.py
   ```

## Requirements
The project dependencies are listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- The dataset used in this project is sourced from publicly available fake news datasets.
- Special thanks to the open-source community for providing tools and libraries used in this project.