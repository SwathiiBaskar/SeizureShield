# SeizureShield
**Seizure Detection & Alert System using EEG and Deep Learning**

SeizureShield is an intelligent system that analyzes **EEG signals** and detects seizure-related patterns using a **Feedforward Neural Network (FNN)** model.  
When abnormal activity is detected, the system automatically sends an **alert via WhatsApp (Twilio API)** and **Firebase Cloud Messaging (FCM)** to registered caregivers.

---

## Project Status
- Software completed  
- Hardware integration in progress  

---

## Tech Stack
- **Backend**: Python, Flask  
- **Machine Learning**: TensorFlow / Keras (FNN)  
- **Database & Notifications**: Firebase Firestore + Firebase Cloud Messaging  
- **Messaging**: Twilio WhatsApp API  
- **Frontend**: HTML, CSS, Chart.js  

---

## Goal
Provide **real-time seizure monitoring** and **timely caregiver alerts** to support epilepsy patients.

---

## Features
- **EEG Signal Analysis** – preprocessing & pattern detection using FNN model  
- **Seizure Probability Prediction** – generates seizure likelihood in real-time  
- **Automatic Caregiver Alerts** – via WhatsApp (Twilio API) & FCM  
- **Cloud Integration** – logs predictions & alerts in Firebase Firestore  
- **Web Dashboard** – built with Flask + Chart.js to monitor alerts & history  
- **Future Scope** – hardware integration with wearable EEG device  

---

## System Architecture

EEG Data → FNN Model → Flask Backend → Firebase + Twilio → Caregiver Alerts → Web Dashboard

---

## Future Scope
- Integration with a **wearable EEG device**  
- Dedicated **mobile app for caregivers**    
- Integration with **electronic medical record systems**  

---

## License
This project is licensed under the **Apache 2.0 License**. 
