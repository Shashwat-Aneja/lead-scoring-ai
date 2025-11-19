# Lead Scoring AI

A machine learning model that predicts the probability of a lead converting into a customer using historical business data.  
Useful for sales optimization and decision-making in business environments.

---

## 📊 Features
- Uses Logistic Regression for binary classification
- Predicts conversion likelihood (0–100%)
- Based on metrics like customer interaction, visits, and spend
- Lightweight, business-friendly, and easily extendable

---

## 🧪 Example CSV Format
```
interaction_score,visits,total_spent,converted
80,10,12000,1
30,4,3000,0
60,7,7000,1
10,2,1000,0
90,12,20000,1
```

---

## 🚀 Usage

### 1. Clone repo
```bash
git clone https://github.com/Shashwat-Aneja/lead-scoring-ai
cd lead-scoring-ai
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn
```

### 3. Run the script
```bash
python score_leads.py data.csv
```

---

## 📎 Project Structure
```
lead-scoring-ai/
│
├── score_leads.py
└── README.md
```

---

## 🔧 Future Enhancements
- Add more features (time since last contact, customer type)
- Use more advanced models (XGBoost / Deep Learning)
- Integration with business CRM systems
- Export results to CSV / PDF

---

Developed by **Shashwat Aneja**
