# sentiment-analysis
Sentimental analysis flask api.
<p align="center">
    <img alt="Open Source" src="https://img.shields.io/badge/Maintained%3F-Yes-9cf?style=for-the-badge">
</p>

## About model

According to analysis, best algorithm for the given problem was Linear SVC. It will fit to the data provided, returning a "best fit" hyperplane that divides, or categorizes, the data.


### How to run the model

```
python3 model.py
```
This will give `classifier.sav` file and `vectorizer.sav` file.

### Running the api

```
python3 api.py

The directory path of api will be http://127.0.0.1:5000/analysis?word=<your-input>

```