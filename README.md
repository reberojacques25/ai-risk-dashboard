# ⚠️ AI Risk Dashboard — Multi-Sector Bias Auditor

**Author:** [Your Name]  
**Project Type:** AI Safety Portfolio — Astra Fellowship (Constellation Institute)  
**Stack:** Python · Streamlit · pandas · scikit-learn · matplotlib · seaborn  
**Live Demo:** [your-app-link.streamlit.app] ← add after deployment

---

## 📌 What This Is

A multi-sector AI bias auditing tool that detects algorithmic discrimination
in high-stakes AI decision systems across three domains:

| Sector | AI System Audited | Stakes |
|--------|-------------------|--------|
| 🏥 **Healthcare** | Patient specialist referral triage | Missed diagnosis, disease progression |
| 💳 **Credit & Lending** | Loan approval scoring | Locked out of homes, education, business |
| ⚖️ **Criminal Justice** | Recidivism risk (detention recommendation) | Loss of freedom, family separation |

The dashboard is designed for **non-technical audiences** — policymakers, compliance teams,
journalists, and civil society — who need to understand AI risk without a data science background.

---

## 🎯 Core Features

### 📊 Bias Audit
- Disparate Impact analysis per protected group (EEOC 4/5ths Rule)
- Plain-English explanations — no jargon, just clear harm descriptions
- Per-group outcome rate charts and AI score distributions
- Color-coded risk levels: ✅ Fair → ⚠️ Borderline → 🚨 Biased → 🔴 Severely Biased

### 🔍 Data Quality Audit
- Missing value detection
- Outlier analysis (IQR method)
- Class imbalance detection and interpretation

### 📋 Downloadable Risk Report
- Auto-generated plain-text audit report
- Specific, prioritized recommendations (Immediate / Short-term / Long-term)
- Downloadable as `.txt` — shareable with compliance teams and regulators

### 🎛️ Overall Bias Risk Score (0–100)
- Single score synthesizing all group-level disparate impact findings
- Displayed as a visual gauge with clear risk level label

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-risk-dashboard.git
cd ai-risk-dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Key Findings (Sample)

### Healthcare
| Group | Referral Rate | Disparate Impact | Status |
|-------|-------------|-----------------|--------|
| Insured | 62.3% | 1.00 (ref) | ⚡ Reference |
| Underinsured | 48.1% | 0.77 | 🚨 Biased |
| Uninsured | 37.4% | 0.60 | 🔴 Severely Biased |
| Black | 41.2% | 0.66 | 🔴 Severely Biased |

### Credit & Lending
| Group | Approval Rate | Disparate Impact | Status |
|-------|-------------|-----------------|--------|
| White | 68.4% | 1.00 (ref) | ⚡ Reference |
| Hispanic | 54.3% | 0.79 | 🚨 Biased |
| Black | 51.7% | 0.76 | 🚨 Biased |

### Criminal Justice
| Group | Detention Rate | Disparate Impact | Status |
|-------|--------------|-----------------|--------|
| White | 38.2% | 1.00 (ref) | ⚡ Reference |
| Hispanic | 52.4% | 1.37 | 🔴 Severely Biased |
| Black | 58.1% | 1.52 | 🔴 Severely Biased |

---

## 🧠 AI Safety Connection

This project directly addresses three core AI safety concerns:

**1. Scalable Harm**  
A biased human recruiter or loan officer affects hundreds of people per year.
A biased AI system affects millions — silently, automatically, at scale.

**2. Invisible Discrimination**  
AI systems produce discrimination that is harder to detect, challenge, or appeal
than human discrimination. Without active auditing tools like this, it goes unseen.

**3. Feedback Loops**  
Biased outputs (e.g., fewer Black patients referred) → less treatment data for Black patients
→ future AI trained on this data → more bias. The loop tightens over time.

---

## 🌍 Africa & Global South Context

AI decision systems are being deployed rapidly across Africa in healthcare, fintech,
and public administration — often adapted from Western models without local fairness auditing.
This dashboard was designed to be:

- **Accessible** to non-technical regulators and civil society
- **Deployable** in low-resource environments (no GPU required)
- **Relevant** to African policy contexts (insurance coverage, income-based discrimination)

---

## 📚 References

- [Fairness and Machine Learning — Barocas, Hardt, Narayanan](https://fairmlbook.org/)
- [EEOC Adverse Impact Guidelines](https://www.eeoc.gov/)
- [AI Fairness 360 — IBM Research](https://aif360.mybluemix.net/)
- [COMPAS Recidivism Algorithm — ProPublica Investigation](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [Anthropic's AI Safety Research](https://www.anthropic.com/research)

---

*Built as part of an AI Safety portfolio for the Astra Fellowship — Constellation Institute, Berkeley.*
