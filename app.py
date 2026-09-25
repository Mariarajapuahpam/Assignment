from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import re


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "agricultural_price_model.pkl"
FEATURE_FILE = "model_features.pkl"
RAG_FILE = "agricultural_price_knowledge.csv"


# ============================================================
# LOAD MODEL, FEATURES AND RAG DATA
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    final_features = joblib.load(FEATURE_FILE)

    rag_data = pd.read_csv(RAG_FILE)

    print("Model loaded successfully")
    print("Feature file loaded successfully")
    print("RAG data loaded successfully")

except Exception as e:
    print("Error while loading project files:")
    print(e)

    model = None
    final_features = None
    rag_data = pd.DataFrame()


# ============================================================
# DATA PREPARATION
# ============================================================

if not rag_data.empty:

    # Convert date column
    if "date" in rag_data.columns:
        rag_data["date"] = pd.to_datetime(
            rag_data["date"],
            errors="coerce"
        )

    elif "t" in rag_data.columns:
        rag_data["date"] = pd.to_datetime(
            rag_data["t"],
            errors="coerce"
        )

    # Convert price columns to numeric
    for column in ["p_min", "p_max", "p_modal"]:

        if column in rag_data.columns:

            rag_data[column] = pd.to_numeric(
                rag_data[column],
                errors="coerce"
            )

    # Remove invalid rows
    required_columns = [
        "p_min",
        "p_max",
        "p_modal"
    ]

    rag_data = rag_data.dropna(
        subset=[
            column for column in required_columns
            if column in rag_data.columns
        ]
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_price(market, date):

    if model is None:
        raise Exception("ML model could not be loaded.")

    if rag_data.empty:
        raise Exception("RAG dataset is empty.")

    date = pd.to_datetime(date)

    # --------------------------------------------------------
    # Calculate input features
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "p_min": [
            rag_data["p_min"].mean()
        ],

        "year": [
            date.year
        ],

        "month": [
            date.month
        ],

        "day": [
            date.day
        ],

        "day_of_week": [
            date.dayofweek
        ],

        "week_of_year": [
            date.isocalendar().week
        ],

        "is_weekend": [
            1 if date.dayofweek >= 5 else 0
        ],

        "quarter": [
            date.quarter
        ],

        "price_range": [
            rag_data["p_max"].mean()
            -
            rag_data["p_min"].mean()
        ],

        "price_average": [
            (
                rag_data["p_min"].mean()
                +
                rag_data["p_max"].mean()
            ) / 2
        ],

        "lag_1": [
            rag_data["p_modal"].iloc[-1]
        ],

        "lag_2": [
            rag_data["p_modal"].iloc[-2]
        ],

        "lag_3": [
            rag_data["p_modal"].iloc[-3]
        ]
    })

    # --------------------------------------------------------
    # Match training features
    # --------------------------------------------------------

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=final_features,
        fill_value=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    return float(prediction)


# ============================================================
# MARKET PRICE RETRIEVAL
# ============================================================

def get_market_price(market):

    if rag_data.empty:
        return None

    market_data = rag_data[
        rag_data["market_name"]
        .astype(str)
        .str.lower()
        ==
        str(market).lower()
    ]

    if market_data.empty:
        return None

    # Sort by date
    market_data = market_data.sort_values(
        "date"
    )

    # Latest available record
    latest = market_data.iloc[-1]

    return {

        "market": str(
            latest.get("market_name", "")
        ),

        "district": str(
            latest.get("district_name", "")
        ),

        "date": str(
            latest["date"].date()
        ),

        "variety": str(
            latest.get("variety", "")
        ),

        "minimum_price": float(
            latest["p_min"]
        ),

        "maximum_price": float(
            latest["p_max"]
        ),

        "modal_price": float(
            latest["p_modal"]
        )
    }


# ============================================================
# RAG CHATBOT
# ============================================================

def agricultural_price_chatbot(user_query):

    query = user_query.lower().strip()

    if rag_data.empty:

        return {
            "type": "error",
            "message": "RAG dataset is not available."
        }

    # --------------------------------------------------------
    # PREDICTION QUERY
    # --------------------------------------------------------

    if (
        "predict" in query
        or "predicted" in query
        or "forecast" in query
    ):

        selected_market = None

        # Find market mentioned by user
        for market in rag_data[
            "market_name"
        ].dropna().unique():

            market_name = str(market)

            if market_name.lower() in query:

                selected_market = market_name
                break

        if selected_market is None:

            return {
                "type": "error",
                "message": (
                    "Please mention a valid market name."
                )
            }

        # Try to find date YYYY-MM-DD
        date_match = re.search(
            r"\d{4}-\d{2}-\d{2}",
            query
        )

        if date_match:

            prediction_date = date_match.group()

        else:

            prediction_date = str(
                pd.Timestamp.today().date()
            )

        try:

            predicted_price = predict_price(
                selected_market,
                prediction_date
            )

            return {

                "type": "prediction",

                "market": selected_market,

                "date": prediction_date,

                "predicted_price": round(
                    predicted_price,
                    2
                ),

                "message": (
                    f"Predicted modal price for "
                    f"{selected_market} on "
                    f"{prediction_date} is "
                    f"₹{predicted_price:.2f}"
                )
            }

        except Exception as e:

            return {
                "type": "error",
                "message": str(e)
            }

    # --------------------------------------------------------
    # HIGHEST PRICE
    # --------------------------------------------------------

    if (
        "highest" in query
        or "maximum price" in query
        or "max price" in query
    ):

        row = rag_data.loc[
            rag_data["p_modal"].idxmax()
        ]

        return {

            "type": "historical",

            "market": str(
                row["market_name"]
            ),

            "date": str(
                row["date"].date()
            ),

            "price": float(
                row["p_modal"]
            ),

            "message": (
                f"The highest modal price is "
                f"₹{row['p_modal']:.2f} at "
                f"{row['market_name']}."
            )
        }

    # --------------------------------------------------------
    # LOWEST PRICE
    # --------------------------------------------------------

    if (
        "lowest" in query
        or "minimum price" in query
        or "min price" in query
    ):

        row = rag_data.loc[
            rag_data["p_modal"].idxmin()
        ]

        return {

            "type": "historical",

            "market": str(
                row["market_name"]
            ),

            "date": str(
                row["date"].date()
            ),

            "price": float(
                row["p_modal"]
            ),

            "message": (
                f"The lowest modal price is "
                f"₹{row['p_modal']:.2f} at "
                f"{row['market_name']}."
            )
        }

    # --------------------------------------------------------
    # AVERAGE PRICE
    # --------------------------------------------------------

    if (
        "average" in query
        or "mean price" in query
    ):

        average_price = rag_data[
            "p_modal"
        ].mean()

        return {

            "type": "average",

            "price": round(
                float(average_price),
                2
            ),

            "message": (
                f"The average modal price is "
                f"₹{average_price:.2f}."
            )
        }

    # --------------------------------------------------------
    # MARKET-SPECIFIC QUERY
    # --------------------------------------------------------

    for market in rag_data[
        "market_name"
    ].dropna().unique():

        market_name = str(market)

        if market_name.lower() in query:

            result = get_market_price(
                market_name
            )

            if result:

                return {

                    "type": "market",

                    **result,

                    "message": (
                        f"Latest price at "
                        f"{result['market']} is "
                        f"₹{result['modal_price']:.2f} "
                        f"(modal price)."
                    )
                }

    # --------------------------------------------------------
    # DEFAULT RESPONSE
    # --------------------------------------------------------

    return {

        "type": "help",

        "message": (
            "Sorry, I could not find the requested "
            "information. You can ask questions such as:\n\n"
            "• What is the price in Udumalpet?\n"
            "• What is the highest price?\n"
            "• What is the lowest price?\n"
            "• What is the average price?\n"
            "• What is the predicted price in Udumalpet?"
        )
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        market = data.get(
            "market"
        )

        date = data.get(
            "date"
        )

        if not market or not date:

            return jsonify({
                "success": False,
                "message": (
                    "Market and date are required."
                )
            }), 400

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

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ============================================================
# CHAT API
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json()

        user_query = data.get(
            "query",
            ""
        )

        if not user_query:

            return jsonify({

                "success": False,

                "message": (
                    "Please enter a question."
                )

            }), 400

        response = agricultural_price_chatbot(
            user_query
        )

        return jsonify({

            "success": True,

            "response": response

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ============================================================
# MARKET LIST API
# ============================================================

@app.route(
    "/markets",
    methods=["GET"]
)
def markets():

    if rag_data.empty:

        return jsonify([])

    market_list = sorted(
        rag_data[
            "market_name"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return jsonify(
        market_list
    )


# ============================================================
# DASHBOARD SUMMARY API
# ============================================================

@app.route(
    "/summary",
    methods=["GET"]
)
def summary():

    if rag_data.empty:

        return jsonify({

            "success": False,

            "message": "No data available."

        })

    return jsonify({

        "success": True,

        "total_records": int(
            len(rag_data)
        ),

        "total_markets": int(
            rag_data[
                "market_name"
            ].nunique()
        ),

        "total_districts": int(
            rag_data[
                "district_name"
            ].nunique()
        ),

        "average_price": round(
            float(
                rag_data[
                    "p_modal"
                ].mean()
            ),
            2
        ),

        "highest_price": round(
            float(
                rag_data[
                    "p_modal"
                ].max()
            ),
            2
        ),

        "lowest_price": round(
            float(
                rag_data[
                    "p_modal"
                ].min()
            ),
            2
        )
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
