# ⚕️ CareGuard AI - Chronic Care Risk Prediction Engine

> **⚠️ IMPORTANT DISCLAIMER:**  
> This application is a **demo/prototype** and is intended for **research and educational purposes only**.  
> All patient data used by CareGuard AI has been **self-generated/synthetic** and does **NOT** represent real individuals or actual clinical records.  
> This tool is **NOT** certified for clinical decision making or medical use on real patients.

🏥 **Advanced Medical Dashboard for Chronic Care Risk Prediction**

CareGuard AI is an interactive web application that predicts if chronic care patients (diabetes, heart failure, hypertension) will deteriorate in the next 90 days. It features a beautiful medical-themed interface with patient registration, risk analytics, and explainable predictions.

## 🌐 Live Demo

**Try the live app:** [https://ashish-1506-careguard-ai.streamlit.app/](https://ashish-1506-careguard-ai.streamlit.app/)

## 🎯 Overview

**Problem**: Chronic patients often deteriorate silently between visits, leading to preventable hospitalizations and poor outcomes.

**Solution**: CareGuard AI uses machine learning to identify high-risk patients early, with transparent explanations and specific clinical recommendations.

**Features**: Interactive dashboard, patient registration system, risk prediction, population analytics, and explainable AI insights.

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Ashish-1506/CareGuard-AI.git
cd CareGuard-AI

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Key Features

### 🏥 Clinical Dashboard
- **Live system statistics** with patient cohort metrics
- **Population health analytics** with risk distribution charts
- **Real-time risk assessment** for proactive care management

### 📋 Patient Registration System
- **Comprehensive patient intake** with full medical history
- **Demographic and clinical data collection** 
- **Persistent patient storage** for ongoing monitoring
- **Integration with risk analysis** workflows

### 🔬 Quick Risk Testing Laboratory
- **Instant risk predictions** without patient registration
- **Tabbed input interface** for clinical data entry
- **Real-time 90-day deterioration risk scoring**

### 👤 Individual Patient Analysis
- **Detailed patient risk assessment** with SHAP explanations
- **Clinical trend visualization** over time
- **Risk factor breakdowns** with impact analysis
- **Personalized clinical recommendations**

### 📊 Population Health Analytics
- **Cohort filtering** by diagnosis, risk level, and demographics
- **Risk distribution visualizations**
- **Clinical alert monitoring** (overdue labs, high-risk patients)
- **Interactive patient registry** with search and sorting

### 📈 System Performance Monitoring
- **Model validation metrics** (AUROC, AUPRC, Sensitivity, Specificity)
- **Clinical utility assessment** with confusion matrices
- **System architecture overview** and configuration details

## 🎨 User Interface

### Medical-Themed Design
- **Healthcare-inspired color palette** with medical blues and clinical whites
- **Glass morphism effects** with professional shadows and gradients  
- **Animated medical elements** with floating particles and pulse effects
- **Responsive card-based layout** optimized for clinical workflows

### Navigation
- **6 main tabs** for different workflows:
  - 🏥 Clinical Dashboard - System overview and live metrics
  - 📊 Population Health - Cohort analysis and filtering  
  - 🔬 Quick Risk Testing - Instant risk assessment
  - 📋 Add New Patient - Comprehensive patient registration
  - 👤 Patient Analysis - Individual patient deep-dive
  - 📈 System Performance - Model metrics and validation

## 📁 Project Structure

```
CareGuard-AI/
├── streamlit_app.py        # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── features.py        # Feature engineering
│   ├── explain.py         # SHAP explainability
│   └── api.py            # FastAPI backend (optional)
├── models/
│   └── model.pkl         # Trained model bundle
├── data/
│   └── processed/
│       └── training_table.csv  # Sample training data
└── assets/               # Screenshots and images
```

## 🔬 Data Features

### Patient Demographics
- Age, sex, primary chronic condition
- Contact information and emergency contacts
- Medical history and family history

### Clinical Measurements  
- **HbA1c**: Diabetes control (last value)
- **Weight trend**: kg change over 30 days
- **Blood pressure**: Systolic and diastolic readings
- **BNP**: Heart failure biomarker
- **eGFR trend**: Kidney function change over 90 days
- **BMI**: Body mass index (auto-calculated)
- **Heart rate**: Current heart rate

### Behavioral & Treatment Factors
- **Medication adherence**: Treatment compliance rate
- **Care engagement**: Days since last lab work
- **Smoking status**: Current smoking habits
- **Exercise frequency**: Physical activity level
- **Diet compliance**: Nutritional adherence

### Clinical Notes
- **Current medications**: List of prescriptions
- **Known allergies**: Drug and food allergies
- **Clinical observations**: Additional symptoms and notes
- **Risk factors**: Other relevant clinical considerations

## 📊 Model Performance

- **AUROC**: 0.85+ (Excellent discrimination)
- **AUPRC**: 0.70+ (Good precision-recall balance)  
- **Risk Bands**: Low (<10%), Medium (10-25%), High (>25%)
- **Calibration**: Isotonic calibration for reliable probabilities

## 🏥 Clinical Use Cases

### Primary Care
- **Population health management** - identify high-risk patients
- **Care coordination** - prioritize interventions
- **Preventive care** - proactive patient outreach

### Chronic Disease Management  
- **Diabetes care** - HbA1c and adherence monitoring
- **Heart failure** - early decompensation detection
- **Hypertension** - medication optimization

### Healthcare Teams
- **Nurse care managers** - patient prioritization
- **Clinical pharmacists** - adherence interventions  
- **Specialist referrals** - optimization and coordination

## 🛡️ Safety & Ethics

### Clinical Safety
- **Decision support tool** - not a replacement for clinical judgment
- **Human oversight required** - designed for clinical assistance only
- **Prototype status** - not validated for real patient care

### Data Privacy & Security
- **Synthetic data only** - no real patient information used
- **Session-based storage** - patient data not permanently stored
- **Educational purpose** - designed for demonstration and learning

### Responsible Development
- **Transparent methodology** - open source codebase
- **Explainable predictions** - SHAP-based interpretability
- **Bias awareness** - designed with fairness considerations

## 🚀 Deployment Options

### Streamlit Community Cloud (Current)
The live demo is deployed on Streamlit Community Cloud:
- **URL**: https://ashish-1506-careguard-ai.streamlit.app/
- **Automatic updates** from GitHub repository
- **Free hosting** for public demonstration

### Local Development
```bash
# Run locally
streamlit run streamlit_app.py
```

### Alternative Platforms
- **Heroku**: For scalable deployment
- **Railway/Render**: For modern cloud hosting
- **Docker**: For containerized deployment

## 🧪 Testing the Application

### Basic Functionality Test
1. **Navigate to tabs** - Ensure all 6 tabs load correctly
2. **Add new patient** - Test the registration form
3. **Quick risk test** - Try the risk assessment lab
4. **Patient analysis** - Select and analyze existing patients
5. **Population health** - Apply filters and view analytics

### Sample Test Data
For testing, try entering:
- **Age**: 65, **Sex**: M, **Condition**: Diabetes
- **HbA1c**: 8.5%, **BP**: 150/90, **BMI**: 30
- **Adherence**: 75%, **Weight trend**: +2kg/month
- **Days since labs**: 120 days

## 👨‍💻 Author & Development

**CareGuard AI** is developed and maintained by **Ashish**.

- **GitHub**: [Ashish-1506](https://github.com/Ashish-1506)
- **Project Repository**: [CareGuard-AI](https://github.com/Ashish-1506/CareGuard-AI)
- **Live Demo**: [Streamlit App](https://ashish-1506-careguard-ai.streamlit.app/)

### Technical Stack
- **Frontend**: Streamlit with custom CSS/HTML
- **Backend**: Python with pandas, numpy, scikit-learn
- **ML Framework**: XGBoost with SHAP explainability
- **Visualization**: Plotly for interactive charts
- **Deployment**: Streamlit Community Cloud

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Development Setup
```bash
# Fork the repository on GitHub
git clone https://github.com/yourusername/CareGuard-AI.git
cd careguard-ai

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test locally
streamlit run streamlit_app.py

# Submit pull request
```

### Areas for Contribution
- **UI/UX improvements** - enhance the medical interface
- **Additional features** - new clinical capabilities
- **Performance optimization** - faster predictions and rendering
- **Documentation** - improve guides and examples
- **Testing** - add automated tests and validation

## 📄 License & Usage

This project is open source and available under the MIT License.

### Usage Rights
- ✅ **Educational use** - learning and research purposes
- ✅ **Personal projects** - individual development and testing
- ✅ **Open source contributions** - improvements and modifications
- ❌ **Commercial medical use** - not validated for clinical practice
- ❌ **Real patient data** - only use synthetic/demo data

## 📞 Support & Contact

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check this README and code comments

### Project Links
- **Live Demo**: [https://ashish-1506-careguard-ai.streamlit.app/](https://ashish-1506-careguard-ai.streamlit.app/)
- **Source Code**: [https://github.com/Ashish-1506/CareGuard-AI](https://github.com/Ashish-1506/CareGuard-AI)
- **Author Profile**: [https://github.com/Ashish-1506](https://github.com/Ashish-1506)

## 🔮 Future Enhancements

### Planned Features
- **Real-time data integration** - API connections to healthcare systems
- **Advanced visualizations** - interactive patient journey maps  
- **Multi-language support** - internationalization for global use
- **Mobile optimization** - responsive design for tablets and phones
- **User authentication** - secure login and patient data protection

### Research Directions
- **Additional chronic conditions** - expand beyond diabetes/heart failure
- **Time-series forecasting** - predict risk trajectories over time
- **Intervention modeling** - simulate treatment effectiveness
- **Population health insights** - community-level analytics

---

**⚕️ CareGuard AI** - Advancing healthcare through intelligent risk prediction and clinical decision support

*Built with precision by Ashish | Powered by Streamlit, XGBoost, and SHAP*