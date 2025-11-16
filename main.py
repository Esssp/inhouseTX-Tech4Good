from model.train import train_model
from model.inference import predict_single
from utils.config import load_taxonomy

if __name__ == "__main__":
    print("Loading taxonomy...")
    taxonomy = load_taxonomy()

    print("Training model...")
    model, tfidf = train_model()

    print("Sample prediction:")
    text = "Starbucks Coffee"
    category, confidence = predict_single(text, model, tfidf)
    print("Input:", text)
    print("Predicted Category:", category)
    print("Confidence:", confidence)
