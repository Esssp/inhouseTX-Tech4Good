import shap
from utils.preprocess import normalise_text

def explain_prediction(text, model, tfidf):
    clean = normalise_text(text)
    vec = tfidf.transform([clean])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(vec)
  
    return shap_values[0].tolist()
