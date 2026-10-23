# Animal Detection AI — Streamlit

This app accepts an animal image and uses a trained TensorFlow/Keras CNN model to predict a class.

## Files needed

- `app.py` — Streamlit website
- `requirements.txt` — dependencies
- `animal_classifier.keras` — **your trained model from Kaggle**
- `class_names.json` — class names in the same order used by the trained model

The included `class_names.json` is set to:
`["Cats", "Dogs", "Elephants", "Lion", "Tigers"]`

## 1. Add your trained model

Download `animal_classifier.keras` from Kaggle and put it in the same folder as `app.py`.
Do not upload the original dataset unless you specifically need it.

## 2. Upload files to GitHub

Create a repository, for example `animal-detection-ai`, and upload:
- `app.py`
- `requirements.txt`
- `class_names.json`
- `animal_classifier.keras`

If the model file exceeds GitHub's standard 100 MB file limit, use Git LFS or suitable external model storage instead.

## 3. Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/
2. Sign in with GitHub.
3. Choose **Create app**.
4. Select your repository and branch (`main`).
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.
7. When deployment finishes, copy the generated `.streamlit.app` URL and send it to your teacher.

## Important notes

- The model must be the actual trained `.keras` file. This project ZIP does not contain your Kaggle-trained model.
- `class_names.json` must match the output class order of the trained model.
- Your current training dataset reportedly has only 10 images, so predictions may be unreliable. Add more labelled images and retrain for a meaningful model.
