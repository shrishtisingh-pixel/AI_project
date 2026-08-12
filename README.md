# ShopFlow AI 📊🤖

ShopFlow AI is a high-performance retail analytics web application combining **Predictive AI** (RandomForestRegressor trained on historical factors) and **Conversational AI** (a chatbot grounded directly on live SQLite prediction logs). 

Managers can simulate traffic patterns, view operational staffing recommendations, inspect explainability drivers, and chat with an assistant regarding dashboard insights.

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Ensure you have Python 3.10+ installed. Clone or navigate to the project directory and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Synthetic Historical Data
Create two years of weekly historical store traffic records featuring seasonal, weather, and economic variations:

```bash
python data/generate_sample_data.py
```
*This generates a file at `data/retail_traffic.csv` containing 1,040 simulated entries across 10 stores.*

### 3. Train the Predictive Model
Train the Random Forest model and extract evaluation metrics and feature importances:

```bash
python ml/train_model.py
```
*This splits the dataset 80/20, trains the `RandomForestRegressor`, outputs evaluation logs (MAE, RMSE, R² score), and saves the trained model to `ml/model/model.joblib` and feature importances to `ml/model/feature_meta.json`.*

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env` and optionally set your Anthropic Claude API Key:

```bash
cp .env.example .env
```

Open `.env` and configure:
```env
ANTHROPIC_API_KEY=your-actual-anthropic-key-here
PORT=8000
HOST=127.0.0.1
```
> [!NOTE]
> **No API Key? No Problem!** If the `ANTHROPIC_API_KEY` is left blank, the app runs in **Offline Simulator Mode** which queries the SQLite predictions log and answers questions using a localized analysis script. You can still test end-to-end functionality!

### 5. Launch the Server
Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```
Once started, open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🛠️ Tech Stack & Hyperparameters
- **Predictive Model**: `RandomForestRegressor` (`n_estimators=200`, `max_depth=10`, `random_state=42`)
- **Train/Test Split**: 80% / 20%
- **Backend Framework**: FastAPI (python-dotenv, Pydantic v2)
- **Database**: SQLite3
- **Frontend**: Single Page Dashboard built with HTML5, vanilla CSS (with glassmorphism, responsive grid, dynamic progress bars) and vanilla JavaScript.
- **Conversational Engine**: Anthropic Claude API (model `claude-3-5-sonnet-20241022`)

---

## 💬 Verification Workflow

1. **Dashboard Prediction**:
   Adjust parameters (e.g., Select Store 3, Date 2026-11-27, toggle Holiday Surge, adjust sliders) and click **Generate Prediction**.
   *Verify that a predicted weekly traffic count up animates, a color-coded traffic badge glows, operational recommendations display, and the top 3 drivers animate their weights.*
2. **SQLite History**:
   Look at the "Predictions History Log" table in the bottom right corner.
   *Verify that your prediction was written to SQLite and displays in the table.*
3. **Conversational Grounding**:
   Click on the suggestion chip: **"Which store had the highest predicted traffic?"** or type **"Why is Store 3 high on Nov 27?"**.
   *Verify that the AI assistant retrieves predictions from SQLite, references them, and replies with grounded calculations.*
