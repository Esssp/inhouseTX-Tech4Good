InHouseTX – Automated AI-Based Financial Transaction Categorisation
Team Tech4Good | GHCI 2025 Hackathon
InHouseTX is an autonomous, locally executed machine-learning system designed to categorise raw financial transaction strings such as “Starbucks Coffee”, “Amazon”, and “Shell Fuel” into meaningful spending categories.
The system operates entirely without third-party APIs, offering full data privacy, reduced cost, and complete customisability. It includes an end-to-end pipeline covering data processing, feature extraction, model training, explainability, confidence scoring, configuration management, and a feedback mechanism for continuous improvement.

- Project Features:
1. End-to-end machine-learning workflow handled within the local environment.
2. Customisable taxonomy configured through "taxonomy.json".
3. Support for TF-IDF with LightGBM or MiniLM embeddings.
4. Integrated explainability using SHAP or LIME.
5. Feedback loop for capturing and learning from low-confidence predictions.
6. REST API built with Flask for both individual and batch inference.
7. Evaluation using macro F1 score, per-class analysis, and confusion matrix.
8. Modular, transparent, and production-friendly structure suitable for scaling.

- Project Structure:
  inhouseTX/
│── README.md
│── dataset.csv
│── taxonomy.json
│── feedback.csv
│── requirements.txt
│── main.py
│
├── api/
│   └── app.py
│
├── utils/
│   ├── preprocess.py
│   ├── explain.py
│   └── config.py
│
├── model/
│   ├── train.py
│   └── inference.py
│
└── diagrams/
    ├── architecture.png
    └── pipeline.png

- System Architecture:
  flowchart TD
  A[Raw Transaction Input] --> B[Pre-processing<br/>Cleaning, Normalisation, Abbreviation Expansion]
  B --> C[Feature Extraction<br/>TF-IDF or MiniLM Embeddings]
  C --> D[Classifier<br/>LightGBM or Dense Neural Network]
  D --> E[Output<br/>{Category, Confidence}]
  D --> F[Explainability<br/>SHAP Token Importance]
  E --> H{Confidence Check}
  H -->|Low Confidence| I[Feedback Queue]
  I --> J[Human Review and Correction]
  J --> K[Retraining Pipeline]
  K --> D
  L[Config Manager<br/>(taxonomy.json)] --> B
  L --> D

- Installation:
pip install -r requirements.txt
python model/train.py
python api/app.py

- Evaluation:
The model produces comprehensive evaluation results, including:
1. Macro F1 score
2. Per-class F1 scores
3. Precision and recall statistics
4. Confusion matrix
5. Optional latency and throughput measurements
All evaluation outputs are generated through model/train.py.

- Configurable Taxonomy:
The categorisation structure is defined entirely through the taxonomy.json file.
New categories can be introduced, modified, or removed without editing the source code.
This allows organisations to align categorisation with internal financial or reporting structures.

- Important Links:
1. Project PDF (to be uploaded)
2. Demonstration video link (to be uploaded)
3. Dataset documentation included in the repository

- Licence:
This project is distributed under the MIT Licence.

- Acknowledgements:
Developed by Team Tech4Good for the GHCI 2025 Hackathon, demonstrating responsible, explainable, and customisable AI for financial-transaction categorisation.
