from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# =========================================================
# LOAD MODEL, FEATURES AND DATA
# =========================================================

try:
    model = joblib.load("agricultural_price_model.pkl")
    final_features = joblib.load("model_features.pkl")
    rag_data = pd.read_csv("agricultural_price_knowledge.csv")

    print("✅ Model loaded successfully")
    print("✅ Features loaded successfully")
    print("✅ Dataset loaded successfully")
    print("Dataset shape:", rag_data.shape)
    print("Model features:", final_features)

except Exception as e:
    print("❌ ERROR WHILE LOADING FILES:", repr(e))
    model = None
    final_features = []
    rag_data = pd.DataFrame()


# =========================================================
# FIND MARKET COLUMN
# =========================================================

def get_market_column():

    possible_columns = [
        "market",
        "Market",
        "market_name",
        "Market Name",
        "Market_Name"
    ]

    for col in possible_columns:
        if col in rag_data.columns:
            return col

    return None


# =========================================================
# FIND DATE COLUMN
# =========================================================

def get_date_column():

    possible_columns = [
        "date",
        "Date",
        "arrival_date",
        "Arrival Date",
        "arrival_date"
    ]

    for col in possible_columns:
        if col in rag_data.columns:
            return col

    return None


# =========================================================
# PRICE PREDICTION FUNCTION
# =========================================================

def predict_price(market, date):

    if model is None:
        raise Exception("Model could not be loaded.")

    if rag_data.empty:
        raise Exception("Agricultural price dataset is empty.")

    # Convert date
    selected_date = pd.to_datetime(date)

    market_column = get_market_column()
    date_column = get_date_column()

    # -----------------------------------------------------
    # FILTER MARKET DATA
    # -----------------------------------------------------

    market_data = rag_data.copy()

    if market_column is not None:

        matching_data = market_data[
            market_data[market_column].astype(str).str.strip().str.lower()
            == str(market).strip().lower()
        ]

        if len(matching_data) > 0:
            market_data = matching_data

        else:
            print("⚠️ Market not found exactly:", market)
            print("Using complete dataset for prediction.")

    # -----------------------------------------------------
    # SORT BY DATE
    # -----------------------------------------------------

    if date_column is not None:

        market_data[date_column] = pd.to_datetime(
            market_data[date_column],
            errors="coerce"
        )

        market_data = market_data.sort_values(
            by=date_column
        )

    # -----------------------------------------------------
    # CHECK REQUIRED PRICE COLUMNS
    # -----------------------------------------------------

    required_columns = [
        "p_min",
        "p_max",
        "p_modal"
    ]

    for col in required_columns:

        if col not in market_data.columns:
            raise Exception(
                f"Required column '{col}' not found in dataset."
            )

    # -----------------------------------------------------
    # LAG VALUES
    # -----------------------------------------------------

    if len(market_data) >= 3:

        lag_1 = float(market_data["p_modal"].iloc[-1])
        lag_2 = float(market_data["p_modal"].iloc[-2])
        lag_3 = float(market_data["p_modal"].iloc[-3])

    else:

        # fallback to complete dataset

        lag_1 = float(rag_data["p_modal"].iloc[-1])
        lag_2 = float(rag_data["p_modal"].iloc[-2])
        lag_3 = float(rag_data["p_modal"].iloc[-3])

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "p_min": [
            float(market_data["p_min"].mean())
        ],

        "year": [
            selected_date.year
        ],

        "month": [
            selected_date.month
        ],

        "day": [
            selected_date.day
        ],

        "day_of_week": [
            selected_date.dayofweek
        ],

        "week_of_year": [
            selected_date.isocalendar().week
        ],

        "is_weekend": [
            1 if selected_date.dayofweek >= 5 else 0
        ],

        "quarter": [
            selected_date.quarter
        ],

        "price_range": [
            float(
                market_data["p_max"].mean()
                - market_data["p_min"].mean()
            )
        ],

        "price_average": [
            float(
                (
                    market_data["p_min"].mean()
                    + market_data["p_max"].mean()
                ) / 2
            )
        ],

        "lag_1": [
            lag_1
        ],

        "lag_2": [
            lag_2
        ],

        "lag_3": [
            lag_3
        ]
    })

    # -----------------------------------------------------
    # ADD MARKET
    # -----------------------------------------------------

    if market_column is not None:
        input_data[market_column] = market

    # -----------------------------------------------------
    # CONVERT CATEGORICAL DATA
    # -----------------------------------------------------

    input_data = pd.get_dummies(input_data)

    # -----------------------------------------------------
    # MATCH MODEL TRAINING FEATURES
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=final_features,
        fill_value=0
    )

    print("\n==============================")
    print("Prediction Input")
    print("==============================")
    print(input_data)
    print("Input shape:", input_data.shape)
    print("==============================")

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    return float(prediction)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:
            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        market = data.get("market")
        date = data.get("date")

        if not market:
            return jsonify({
                "success": False,
                "error": "Market is required."
            }), 400

        if not date:
            return jsonify({
                "success": False,
                "error": "Date is required."
            }), 400

        print("\n================================")
        print("PREDICTION REQUEST")
        print("Market:", market)
        print("Date:", date)
        print("================================")

        prediction = predict_price(
            market,
            date
        )

        return jsonify({

            "success": True,

            "market": market,

            "date": date,

            "predicted_price": round(
                prediction,
                2
            )

        })

    except Exception as e:

        # IMPORTANT:
        # This prints the REAL backend error

        print("\n❌ REAL PREDICTION ERROR")
        print(type(e).__name__)
        print(str(e))

        return jsonify({

            "success": False,

            "error": str(e),

            "error_type": type(e).__name__

        }), 500


# =========================================================
# RAG / CHATBOT API
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please enter a question."
            })

        if rag_data.empty:

            return jsonify({
                "answer": "Agricultural price data is not available."
            })

        q = question.lower()

        # -------------------------------------------------
        # HIGHEST PRICE
        # -------------------------------------------------

        if "highest" in q or "maximum" in q or "max" in q:

            row = rag_data.loc[
                rag_data["p_modal"].idxmax()
            ]

            answer = (
                f"The highest modal price in the dataset "
                f"is ₹{float(row['p_modal']):,.2f}."
            )

        # -------------------------------------------------
        # LOWEST PRICE
        # -------------------------------------------------

        elif "lowest" in q or "minimum" in q or "min" in q:

            row = rag_data.loc[
                rag_data["p_modal"].idxmin()
            ]

            answer = (
                f"The lowest modal price in the dataset "
                f"is ₹{float(row['p_modal']):,.2f}."
            )

        # -------------------------------------------------
        # AVERAGE
        # -------------------------------------------------

        elif "average" in q or "mean" in q:

            average_price = rag_data[
                "p_modal"
            ].mean()

            answer = (
                f"The average modal price in the dataset "
                f"is ₹{float(average_price):,.2f}."
            )

        # -------------------------------------------------
        # TOTAL RECORDS
        # -------------------------------------------------

        elif (
            "how many" in q
            or "records" in q
            or "data" in q
        ):

            answer = (
                f"The dataset contains "
                f"{len(rag_data)} records."
            )

        # -------------------------------------------------
        # MARKET INFORMATION
        # -------------------------------------------------

        elif "market" in q:

            market_column = get_market_column()

            if market_column:

                markets = (
                    rag_data[market_column]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                answer = (
                    f"The dataset contains "
                    f"{len(markets)} different markets."
                )

            else:

                answer = (
                    "Market information is not available "
                    "in the dataset."
                )

        # -------------------------------------------------
        # DEFAULT
        # -------------------------------------------------

        else:

            answer = (
                "I can answer questions about agricultural "
                "prices, such as highest price, lowest price, "
                "average price, number of records, and markets."
            )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("\n❌ CHAT ERROR")
        print(type(e).__name__)
        print(str(e))

        return jsonify({

            "answer": (
                f"Chatbot error: {str(e)}"
            )

        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "model_loaded": model is not None,

        "dataset_loaded": not rag_data.empty,

        "dataset_rows": len(rag_data)

    })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("🌾 Agricultural Price Prediction System")
    print("========================================")

    print("Model loaded:", model is not None)

    print(
        "Dataset rows:",
        len(rag_data)
    )

    print(
        "Market column:",
        get_market_column()
    )

    print(
        "Date column:",
        get_date_column()
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
