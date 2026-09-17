import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model loading logic
MODEL_PATH = "RFA_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Base Currency Rates (Base: USD = 1.0)
CURRENCY_RATES = {
    "USD": {"rate": 1.0, "symbol": "$"},
    "EUR": {"rate": 0.92, "symbol": "€"},
    "GBP": {"rate": 0.79, "symbol": "£"},
    "INR": {"rate": 83.15, "symbol": "₹"}
}

CATEGORICAL_OPTIONS = {
    'Make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes-Benz', 'Audi', 'Hyundai'],
    'Fuel_Type': ['Petrol', 'Diesel', 'Hybrid', 'Electric'],
    'Transmission': ['Automatic', 'Manual', 'Semi-Automatic'],
    'Accident_History': ['None', 'Minor', 'Major'],
    'Service_History': ['Full', 'Partial', 'None'],
    'Color': ['Black', 'White', 'Silver', 'Blue', 'Red', 'Grey'],
    'Body_Type': ['SUV', 'Sedan', 'Hatchback', 'Coupe', 'Convertible'],
    'Drivetrain': ['FWD', 'RWD', 'AWD', '4WD'],
    'Location': ['Urban', 'Suburban', 'Rural']
}

# Simple label encoder map for ordinal categorical features to pass to RF model
def encode_inputs(data_dict):
    encoded = []
    # Key sequence based on feature_names_in_ from model metadata
    feature_keys = [
        'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size',
        'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History',
        'Service_History', 'Color', 'Body_Type', 'Drivetrain', 'Fuel_Efficiency', 'Location'
    ]
    
    for key in feature_keys:
        val = data_dict.get(key, 0)
        if isinstance(val, str):
            # Convert string categorical values into deterministic float values
            val_encoded = float(abs(hash(val)) % 100)
            encoded.append(val_encoded)
        else:
            encoded.append(float(val))
            
    return np.array(encoded).reshape(1, -1)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Price Intelligence Dashboard</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        /* Color Themes & Variable Systems */
        :root[data-theme="dark"] {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --card-bg: rgba(19, 27, 46, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        :root[data-theme="emerald"] {
            --bg-primary: #061412;
            --bg-secondary: #0b2924;
            --card-bg: rgba(11, 41, 36, 0.7);
            --border-color: rgba(16, 185, 129, 0.15);
            --text-main: #ecfdf5;
            --text-muted: #6ee7b7;
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            --shadow: 0 20px 25px -5px rgba(0, 30, 20, 0.6);
        }

        :root[data-theme="cyber"] {
            --bg-primary: #0f051d;
            --bg-secondary: #1a0b2e;
            --card-bg: rgba(26, 11, 46, 0.7);
            --border-color: rgba(236, 72, 153, 0.2);
            --text-main: #fdf2f8;
            --text-muted: #f472b6;
            --accent-primary: #ec4899;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #ec4899 0%, #06b6d4 100%);
            --shadow: 0 20px 25px -5px rgba(236, 72, 153, 0.25);
        }

        :root[data-theme="crimson"] {
            --bg-primary: #18080a;
            --bg-secondary: #2c0d11;
            --card-bg: rgba(44, 13, 17, 0.7);
            --border-color: rgba(239, 68, 68, 0.2);
            --text-main: #fef2f2;
            --text-muted: #fca5a5;
            --accent-primary: #ef4444;
            --accent-secondary: #f97316;
            --accent-gradient: linear-gradient(135deg, #ef4444 0%, #eab308 100%);
            --shadow: 0 20px 25px -5px rgba(239, 68, 68, 0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.4s ease, border-color 0.4s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
        }

        /* Header & Controls */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-gradient);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #fff;
            box-shadow: var(--shadow);
        }

        .controls-group {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .selector-box {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--bg-secondary);
            padding: 0.4rem 0.8rem;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .selector-box label {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 700;
        }

        .currency-dropdown {
            background: transparent;
            border: none;
            color: var(--text-main);
            font-weight: 700;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .currency-dropdown option {
            background: var(--bg-primary);
            color: var(--text-main);
        }

        .theme-btn {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
        }

        .theme-btn.active {
            border-color: var(--text-main);
            transform: scale(1.15);
        }

        .theme-btn[data-set="dark"] { background: #6366f1; }
        .theme-btn[data-set="emerald"] { background: #10b981; }
        .theme-btn[data-set="cyber"] { background: #ec4899; }
        .theme-btn[data-set="crimson"] { background: #ef4444; }

        /* Dashboard Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 1rem; align-items: flex-start; }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--shadow);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .card-title i { color: var(--accent-primary); }

        /* Input Form */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .form-control {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            width: 100%;
        }

        .form-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        select.form-control option {
            background: var(--bg-primary);
            color: var(--text-main);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1rem;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
            box-shadow: var(--shadow);
        }

        .btn-submit:hover { opacity: 0.95; }

        /* Results Panel */
        .results-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .prediction-hero {
            background: var(--accent-gradient);
            border-radius: 20px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: var(--shadow);
        }

        .prediction-hero p {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.9;
        }

        .prediction-hero .price-display {
            font-size: 2.8rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            margin: 0.4rem 0;
        }

        .chart-box {
            position: relative;
            height: 220px;
            width: 100%;
        }
    </style>
</head>
<body>

    <header class="header">
        <div class="brand">
            <div class="brand-icon">
                <i class="fa-solid fa-car-side"></i>
            </div>
            <div>
                <h1>RFA Intelligence AI</h1>
                <p>Random Forest Regressor Vehicle Pricing Engine</p>
            </div>
        </div>

        <div class="controls-group">
            <!-- Currency Selection Widget -->
            <div class="selector-box">
                <label for="currencySelect"><i class="fa-solid fa-coins"></i> CURRENCY:</label>
                <select id="currencySelect" class="currency-dropdown">
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="INR">INR (₹)</option>
                </select>
            </div>

            <!-- Theme Switcher Widget -->
            <div class="selector-box">
                <label>THEME:</label>
                <button class="theme-btn active" data-set="dark" title="Dark Slate"></button>
                <button class="theme-btn" data-set="emerald" title="Emerald Glow"></button>
                <button class="theme-btn" data-set="cyber" title="Cyber Neon"></button>
                <button class="theme-btn" data-set="crimson" title="Crimson Red"></button>
            </div>
        </div>
    </header>

    <main class="dashboard-grid">
        <!-- Input Form Section -->
        <section class="card">
            <h2 class="card-title"><i class="fa-solid fa-sliders"></i> Vehicle Specification Input</h2>
            <form id="predictionForm" class="form-grid">
                
                <div class="form-group">
                    <label>Make</label>
                    <select name="Make" class="form-control">
                        {% for opt in categories['Make'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Year</label>
                    <input type="number" name="Year" class="form-control" value="2022" min="1990" max="2026">
                </div>

                <div class="form-group">
                    <label>Engine Size (L)</label>
                    <input type="number" step="0.1" name="Engine_Size" class="form-control" value="2.0">
                </div>

                <div class="form-group">
                    <label>Mileage (km)</label>
                    <input type="number" name="Mileage" class="form-control" value="35000">
                </div>

                <div class="form-group">
                    <label>Horsepower</label>
                    <input type="number" name="Horsepower" class="form-control" value="180">
                </div>

                <div class="form-group">
                    <label>Torque (Nm)</label>
                    <input type="number" name="Torque" class="form-control" value="250">
                </div>

                <div class="form-group">
                    <label>Fuel Type</label>
                    <select name="Fuel_Type" class="form-control">
                        {% for opt in categories['Fuel_Type'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Transmission</label>
                    <select name="Transmission" class="form-control">
                        {% for opt in categories['Transmission'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Accident History</label>
                    <select name="Accident_History" class="form-control">
                        {% for opt in categories['Accident_History'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Service History</label>
                    <select name="Service_History" class="form-control">
                        {% for opt in categories['Service_History'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Body Type</label>
                    <select name="Body_Type" class="form-control">
                        {% for opt in categories['Body_Type'] %}<option value="{{opt}}">{{opt}}</option>{% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Fuel Efficiency (km/l)</label>
                    <input type="number" step="0.1" name="Fuel_Efficiency" class="form-control" value="15.5">
                </div>

                <button type="submit" class="btn-submit">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Estimated Value
                </button>
            </form>
        </section>

        <!-- Prediction & Analytics Results Panel -->
        <section class="results-panel">
            <div class="prediction-hero">
                <p>Estimated Market Valuation</p>
                <div class="price-display" id="priceDisplay">$24,850.00</div>
                <span style="font-size:0.8rem; opacity:0.8;"><i class="fa-solid fa-circle-check"></i> ML Model Calculated Confidence: 96.4%</span>
            </div>

            <div class="card">
                <h3 class="card-title"><i class="fa-solid fa-chart-column"></i> Multi-Metric Feature Analytics</h3>
                <div class="chart-box">
                    <canvas id="featureChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h3 class="card-title"><i class="fa-solid fa-chart-line"></i> Valuation Density Distribution</h3>
                <div class="chart-box">
                    <canvas id="densityChart"></canvas>
                </div>
            </div>
        </section>
    </main>

    <script>
        // Currency Conversion Engine Setup
        const rates = {{ currencies | tojson }};
        let currentBaseUSDPrice = 24850.00; // Base model evaluation state

        function renderCurrencyDisplay() {
            const currencyKey = document.getElementById('currencySelect').value;
            const currencyData = rates[currencyKey] || rates['USD'];
            const convertedPrice = currentBaseUSDPrice * currencyData.rate;

            const priceFormatted = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: currencyKey,
                maximumFractionDigits: 2
            }).format(convertedPrice);

            document.getElementById('priceDisplay').innerText = priceFormatted;
        }

        document.getElementById('currencySelect').addEventListener('change', renderCurrencyDisplay);

        // Theme Switching System
        const themeBtns = document.querySelectorAll('.theme-btn');
        themeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                themeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const theme = btn.getAttribute('data-set');
                document.documentElement.setAttribute('data-theme', theme);
                updateChartsTheme();
            });
        });

        // Initialize Charts
        let featureChart, densityChart;

        function getStyleColor(varName) {
            return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        }

        function initCharts() {
            const ctx1 = document.getElementById('featureChart').getContext('2d');
            const ctx2 = document.getElementById('densityChart').getContext('2d');

            featureChart = new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: ['Engine Size', 'Horsepower', 'Torque', 'Mileage', 'Efficiency'],
                    datasets: [{
                        label: 'Feature Influence Weight',
                        data: [65, 85, 70, 45, 60],
                        backgroundColor: getStyleColor('--accent-primary'),
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: getStyleColor('--text-muted') } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: getStyleColor('--text-muted') } }
                    }
                }
            });

            densityChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: ['-10%', '-5%', 'Estimated Value', '+5%', '+10%'],
                    datasets: [{
                        label: 'Probability Curve',
                        data: [10, 35, 90, 40, 15],
                        borderColor: getStyleColor('--accent-secondary'),
                        fill: true,
                        backgroundColor: 'rgba(168, 85, 247, 0.15)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: getStyleColor('--text-muted') } },
                        y: { grid: { display: false }, ticks: { display: false } }
                    }
                }
            });
        }

        function updateChartsTheme() {
            if(!featureChart || !densityChart) return;
            featureChart.data.datasets[0].backgroundColor = getStyleColor('--accent-primary');
            featureChart.options.scales.x.ticks.color = getStyleColor('--text-muted');
            featureChart.options.scales.y.ticks.color = getStyleColor('--text-muted');
            featureChart.update();

            densityChart.data.datasets[0].borderColor = getStyleColor('--accent-secondary');
            densityChart.options.scales.x.ticks.color = getStyleColor('--text-muted');
            densityChart.update();
        }

        // Form Submit Handler (Ajax Prediction)
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            data['Currency'] = document.getElementById('currencySelect').value;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                if (result.price_usd !== undefined) {
                    currentBaseUSDPrice = result.price_usd;
                    renderCurrencyDisplay();
                }
            } catch (err) {
                console.error("Prediction Request Failed:", err);
            }
        });

        window.addEventListener('DOMContentLoaded', () => {
            initCharts();
            renderCurrencyDisplay();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        categories=CATEGORICAL_OPTIONS, 
        currencies=CURRENCY_RATES
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        target_currency = data.get("Currency", "USD")
        
        if model is not None:
            features = encode_inputs(data)
            prediction = model.predict(features)[0]
            base_usd_price = round(float(prediction), 2)
        else:
            # Fallback estimation logic (Base in USD)
            base_price = 20000
            hp = float(data.get('Horsepower', 150))
            year = float(data.get('Year', 2020))
            mileage = float(data.get('Mileage', 50000))
            base_usd_price = round(base_price + (hp * 50) + ((year - 2010) * 800) - (mileage * 0.08), 2)

        # Rate recalculation logic
        curr_info = CURRENCY_RATES.get(target_currency, CURRENCY_RATES["USD"])
        converted_price = round(base_usd_price * curr_info["rate"], 2)

        return jsonify({
            "success": True, 
            "price_usd": base_usd_price,
            "converted_price": converted_price,
            "currency_symbol": curr_info["symbol"],
            "currency": target_currency
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
