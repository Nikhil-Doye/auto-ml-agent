# 🤖 Auto ML Agent   

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  [![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/)  [![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)](https://streamlit.io/)  [![E2B Sandbox](https://img.shields.io/badge/Execution-E2B_Sandbox-purple)](https://e2b.dev/)  

## 📌 Overview  
**Auto ML Agent** is an **autonomous machine learning pipeline** orchestrated by **LLMs**.  
It automates the **end-to-end ML lifecycle** — from data preprocessing to model training, evaluation, and deployment — with **zero manual intervention**.  

✨ Think of it as your **self-driving data scientist**.  

---

## ⚡ Features  
- 🧹 **Agent 1 – Preprocessing**: Handles missing values, categorical encoding, and normalization.  
- 🏋️ **Agent 2 – Training & Hyperparameter Tuning**: Trains multiple models and optimizes them.  
- 📊 **Agent 3 – Evaluation & Selection**: Compares models on metrics and selects the best one.  
- 🚀 **Agent 4 – Deployment**: Packages the model for inference.  
- 📝 **Natural Language Reports**: LLM generates explanations and insights automatically.  
- 🔒 **Sandboxed Execution**: Safe code execution inside **E2B sandbox**.  

---

## 🏗️ Architecture  

```mermaid
flowchart TD
    A[Autonomous ML Agent (LLM Orchestrator)]
    B[Agent 1 - Preprocessing\nHandle nulls, encode, scale]
    C[Agent 2 - Training & Tuning\nMultiple models + Hyperparams]
    D[Agent 3 - Evaluation & Selection\nBest model chosen]
    E[Agent 4 - Deployment\nPackaged for inference]
    F[LLM Reports\nNatural language explanations]

    A --> B --> C --> D --> E --> F
```

---

## 🚀 Getting Started  

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Nikhil-Doye/auto-ml-agent.git
cd auto-ml-agent
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app
```bash
streamlit run app.py
```

---

## 🧠 How It Works  
1. Upload your **CSV dataset**.  
2. The agent runs **EDA + preprocessing**.  
3. Multiple ML models are trained & tuned.  
4. The best model is automatically selected.  
5. **Reports & explanations** are generated in natural language.  
6. The model can be exported for **deployment**.  

---

## 📊 Example Use Case  
- A business analyst uploads sales data (CSV).  
- Within minutes, the **Auto ML Agent**:  
  - Cleans missing values  
  - Trains models (Random Forest, XGBoost, Logistic Regression, etc.)  
  - Evaluates them with metrics like accuracy, precision, recall  
  - Returns **best model + plain-English explanation**  

---

## 🔮 Future Scope  
- Expand to **unstructured data** (images, text, audio).  
- Add **cloud auto-deployment** (AWS/GCP/Azure).  
- Support **active learning** for continuous improvement.  
- Provide **UI for non-technical users**.  

---

## 🤝 Contributing  
Contributions are welcome!  
1. Fork the repo  
2. Create your feature branch (`git checkout -b feature/awesome-idea`)  
3. Commit changes (`git commit -m 'Add awesome idea'`)  
4. Push to branch (`git push origin feature/awesome-idea`)  
5. Open a Pull Request  

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

## ⭐ Acknowledgements  
- [Streamlit](https://streamlit.io/) – UI Framework  
- [E2B](https://e2b.dev/) – Sandbox environment  
- [OpenAI](https://openai.com/) – LLM orchestration  

---

🔥 If you like this project, **give it a star** on GitHub!  
