# customer-churn-prediction-mlops
End-to-end MLOps pipeline for customer churn prediction with XGBoost, MLflow, Docker, and CI/CD
# Customer Churn Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in--development-yellow.svg)

End-to-end MLOps pipeline for predicting customer churn using the Telco dataset with production-grade tooling.

## 🎯 Project Overview

This project demonstrates a complete machine learning operations (MLOps) workflow from data exploration to deployment, including:

- **Model**: XGBoost classifier for binary churn prediction
- **Experiment Tracking**: MLflow for parameter and metric logging
- **Model Serving**: FastAPI REST API
- **Containerization**: Docker for reproducible deployments
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Streamlit dashboard for model performance tracking

## 🏗️ Architecture
```
Data → Preprocessing → Training → MLflow Tracking → Model Registry
                                        ↓
                                   FastAPI API
                                        ↓
                                   Docker Container
                                        ↓
                                   CI/CD Pipeline
                                        ↓
                                Monitoring Dashboard
```

## 📊 Dataset

- **Source**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle/IBM)
- **Size**: ~7,000 samples
- **Features**: 20 features including customer demographics, services, and account information
- **Target**: Binary classification (Churn: Yes/No)

## 🛠️ Tech Stack

- **ML Framework**: XGBoost, scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Streamlit
- **Testing**: pytest
- **Code Quality**: black, flake8

## 📁 Project Structure
```
customer-churn-mlops/
├── src/
│   ├── preprocessing/    # Data preprocessing modules
│   ├── training/         # Model training scripts
│   ├── serving/          # FastAPI application
│   └── monitoring/       # Streamlit dashboard
├── tests/                # Unit and integration tests
├── notebooks/            # Exploratory data analysis
├── config/               # Configuration files
├── data/                 # Data directory (gitignored)
├── models/               # Saved models (gitignored)
└── .github/workflows/    # CI/CD pipelines
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-mlops.git
   cd customer-churn-mlops
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Download dataset**
```bash
   # Instructions coming soon
```

### Usage

**Training**
```bash
# Coming soon
python src/training/train.py
```

**API**
```bash
# Coming soon
uvicorn src.serving.app:app --reload
```

**Docker**
```bash
# Coming soon
docker-compose up
```

## 📈 Model Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | TBD   |
| F1 Score  | TBD   |
| AUC-ROC   | TBD   |

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📝 Documentation

- [API Documentation](docs/API_DOCUMENTATION.md) - Coming soon
- [Model Card](docs/MODEL_CARD.md) - Coming soon
- [Deployment Guide](docs/DEPLOYMENT.md) - Coming soon

## 🛤️ Roadmap

- [x] Project setup
- [ ] Exploratory Data Analysis
- [ ] Feature Engineering
- [ ] Model Training & Tuning
- [ ] MLflow Integration
- [ ] FastAPI Development
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] Monitoring Dashboard
- [ ] Cloud Deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@vaka2020](https://github.com/vaka2020)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset provided by IBM/Kaggle
- Inspired by industry best practices in MLOps

---

