from utils.preprocess import normalise_text

def predict_single(text, model, tfidf):
    clean = normalise_text(text)
    vec = tfidf.transform([clean])

    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    return prediction, float(confidence)
