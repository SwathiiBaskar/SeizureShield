import time
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from twilio.rest import Client
import threading
from flask import Flask, send_from_directory, render_template, jsonify
import os
from dotenv import load_dotenv

app = Flask(__name__, template_folder="C:/Users/swath/OneDrive/Desktop/Swathi/Projects/ProjectSeizure/ProjectSeizure/templates")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('SeizureShieldFrontend.html', autoescape=True)

#Loading the trained ML model
try:
    with open('seizure_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#Initializing Firebase
try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized!")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

#Twilio configuration
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_PHONE_NUMBER = "+919150373129"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

#Simulated EEG data generation
def generate_eeg_data():
    return np.random.rand(1, 178)

#Predicting seizure probability using the ML model
def predict_eeg():
    if model is None:
        print("Error: Model not loaded.")
        return None

    eeg_data = generate_eeg_data()
    try:
        probabilities = model.predict(eeg_data)
        seizure_probability = float(probabilities[0][0])
        return seizure_probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

#Storing predictions in Firestore
def store_prediction(seizure_probability):
    if db is None:
        print("Firestore not initialized. Skipping storage.")
        return

    db.collection("eeg_predictions").document().set({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "seizure_probability": seizure_probability
    })

#Log high-risk alerts in Firestore
def log_alert(seizure_probability):
    if db is None:
        print("Firestore not initialized. Skipping alert logging.")
        return

    db.collection("alerts").document().set({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "seizure_probability": seizure_probability,
        "alert_sent": True
    })

#Sending WhatsApp alert via Twilio
def send_whatsapp_alert(seizure_probability):
    if seizure_probability is None:
        print("Cannot send alert: Seizure probability is None.")
        return

    try:
        message = twilio_client.messages.create(
            body=f"High seizure probability detected: {seizure_probability:.2f}! Immediate attention required.",
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"WhatsApp alert sent: {message.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp alert: {e}")

#Sending Firebase Cloud Messaging (FCM) notification
def send_fcm_alert(seizure_probability):
    if seizure_probability is None:
        print("Cannot send FCM alert: Seizure probability is None.")
        return

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="Seizure Alert!",
                body=f"High seizure probability detected: {seizure_probability:.2f}. Immediate attention required!"
            ),
            topic="seizure_alerts"
        )
        response = messaging.send(message)
        print(f"FCM Notification sent: {response}")
    except Exception as e:
        print(f"Error sending FCM notification: {e}")

#EEG monitoring loop in a separate thread
def eeg_monitoring():
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 15:
            #After 15 seconds, force seizure probability > 80%
            seizure_probability = 0.85
            print("Simulating Seizure Alert (Forced after 15s)!")
        else:
         seizure_probability = predict_eeg()

        if seizure_probability is not None:
            store_prediction(seizure_probability)

            if seizure_probability > 0.8:
                log_alert(seizure_probability)
                send_whatsapp_alert(seizure_probability)
                send_fcm_alert(seizure_probability)

        time.sleep(5)  # Simulated continuous monitoring

#Starting EEG monitoring in a background thread
if model is not None:
    eeg_thread = threading.Thread(target=eeg_monitoring, daemon=True)
    eeg_thread.start()
    print("EEG monitoring started...")

#Route to fetch seizure alert history
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

@app.route('/seizure_alerts', methods=['GET'])
def get_seizure_alerts():
    if db is None:
        return jsonify({"error": "Firestore not initialized"}), 500

    try:
        alerts_ref = db.collection("alerts").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        alerts = []
        for alert in alerts_ref:
            data = alert.to_dict()
            ts = data.get("timestamp")
            if ts:
                # Convert Firestore Timestamp to ISO string
                timestamp_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            else:
                timestamp_str = "Unknown"
            alerts.append({
                "timestamp": timestamp_str,
                "seizure_probability": data.get("seizure_probability", "N/A")
            })
        return jsonify(alerts)
    except Exception as e:
        print(f"Error retrieving alerts: {e}")
        return jsonify({"error": "Failed to retrieve alerts"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
