from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
from datetime import date as pydate
import json
from pywebpush import webpush, WebPushException

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Firebase Admin SDK (assuming you've set up the credentials)
#cred = credentials.Certificate('path/to/your/serviceAccountKey.json') # Replace with your key file path
cred = credentials.Certificate('credentials/swasthyasathi-firebase-firebase-adminsdk-fbsvc-f4d90b5b72.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# You might want to store these as environment variables instead
VAPID_PUBLIC_KEY = "BHYyXiAoFlAHQ2pAduneylMVM9X0qlQOsT8nBjA7R5TtySi-hDybms8V_XewxPhhUNskxd6lkgylysdU04PYwdw"
VAPID_PRIVATE_KEY = "KEUfRUUFgWGPGlMrSaIlZa8m_RFxMs7sucQMUp6-GNA"
VAPID_CLAIMS = {"sub": "mailto:prasadrajukv@gmail.com", "aud": "https://fcm.googleapis.com"} # Replace with your actual email

@app.route('/api/send_notification', methods=['POST'])
def send_push_notification():
    user_id_to_notify = "test_user" # The same hardcoded user ID used during subscription
    try:
        push_subscriptions_ref = db.collection('PushSubscriptions')
        doc = push_subscriptions_ref.document(user_id_to_notify).get()

        if not doc.exists:
            return jsonify({"error": "No push subscription found for this user."}), 404

        subscription_data = doc.to_dict()

        subscription_info = {
            "endpoint": subscription_data.get('endpoint'),
            "keys": {
                "p256dh": subscription_data.get('keys', {}).get('p256dh'),
                "auth": subscription_data.get('keys', {}).get('auth'),
            },
        }
        message = json.dumps({"title": "Your Reminder!", "body": "Time for your medication."})

        webpush(
            subscription_info=subscription_info,
            data=message,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims=VAPID_CLAIMS
        )
        return jsonify({"success": True}), 200

    except WebPushException as exc:
        print(f"I'm sorry, but the push service said no: {exc}")
        return jsonify({"error": str(exc)}), 400
    except Exception as e:
        print(f"Error sending notification: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/habits', methods=['GET'])
def get_habits():
    habits_collection = db.collection('Habits') # Assuming your collection in Firestore is named 'Habits'
    habits = []
    try:
        docs = habits_collection.stream()
        for doc in docs:
            habit_data = doc.to_dict()
            habits.append({
                'id': doc.id,
                'name_en': habit_data.get('name_en'),
                'name_te': habit_data.get('name_te')
            })
        return jsonify(habits)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/habits/complete', methods=['POST', 'OPTIONS'])
def complete_habit():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    elif request.method == 'POST':
        habit_id = request.json.get('habit_id')
        is_completed = request.json.get('is_completed')
        if not habit_id:
            return jsonify({"error": "Habit ID is required"}), 400

        today = pydate.today().isoformat()
        completion_collection = db.collection('HabitCompletions')
        query = completion_collection.where('habit_id', '==', habit_id).where('completion_date', '==', today)
        docs = query.get()

        try:
            if docs:
                doc_ref = completion_collection.document(docs[0].id)
                doc_ref.update({'is_completed': is_completed})
                return jsonify({"message": f"Habit {habit_id} completion status updated"}), 200
            else:
                completion_collection.add({'habit_id': habit_id, 'completion_date': today, 'is_completed': is_completed})
                return jsonify({"message": f"Habit {habit_id} marked as {'completed' if is_completed else 'not completed'} today"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/habits/status', methods=['POST'])
def get_habit_status():
    habit_ids = request.json.get('habit_ids')
    date_str = request.json.get('date')

    if not habit_ids:
        return jsonify({"error": "Habit IDs are required"}), 400

    if date_str:
        try:
            completion_date = pydate.fromisoformat(date_str)
            date_to_check = completion_date.isoformat()
        except ValueError:
            return jsonify({"error": "Invalid datetime format. Please use formats like YYYY-MM-DD"}), 400
    else:
        today = pydate.today()
        date_to_check = today.isoformat()

    completion_collection = db.collection('HabitCompletions')
    completions = {}

    try:
        for habit_id in habit_ids:
            query = completion_collection.where('habit_id', '==', habit_id).where('completion_date', '==', date_to_check).where('is_completed', '==', True).limit(1)
            docs = query.get()
            completions[habit_id] = not not docs # True if there is at least one doc, False otherwise
        return jsonify(completions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/water/log', methods=['POST'])
def log_water_intake():
    quantity = request.json.get('quantity')
    if quantity is None:
        return jsonify({"error": "Quantity of water is required"}), 400

    timestamp = datetime.now()

    try:
        water_intake_collection = db.collection('WaterIntake')
        water_intake_collection.add({'quantity': quantity, 'timestamp': timestamp})
        return jsonify({"message": f"{quantity}ml of water intake logged successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/water/today', methods=['GET'])
def get_todays_water_intake():
    today = pydate.today().isoformat()
    total_intake = 0

    try:
        water_intake_collection = db.collection('WaterIntake')
        query = water_intake_collection.where('timestamp', '>=', datetime.fromisoformat(today + 'T00:00:00')).where('timestamp', '<=', datetime.fromisoformat(today + 'T23:59:59.999'))
        docs = query.stream()
        for doc in docs:
            water_data = doc.to_dict()
            total_intake += water_data.get('quantity', 0)
        return jsonify({"total_intake_ml": total_intake}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sleep/log', methods=['POST'])
def log_sleep():
    start_time_str = request.json.get('start_time')
    end_time_str = request.json.get('end_time')

    print(f"Received start_time: {start_time_str}")
    print(f"Received end_time: {end_time_str}")

    if not start_time_str or not end_time_str:
        return jsonify({"error": "Start and end times are required"}), 400

    start_time = None
    end_time = None

    formats = ["%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M", "%d/%b/%Y %H:%M"] # Added new formats
    for fmt in formats:
        try:
            start_time = datetime.strptime(start_time_str, fmt)
            break
        except ValueError:
            continue

    for fmt in formats:
        try:
            end_time = datetime.strptime(end_time_str, fmt)
            break
        except ValueError:
            continue

    print(f"Parsed start_time: {start_time}")
    print(f"Parsed end_time: {end_time}")

    if not start_time or not end_time:
        error_message_part1 = "Invalid datetime format. Please use formats like "
        error_message_part2 = "%Y-%m-%dT%H:%M, DD/MM/YYYY HH:MM, or DD-Mon-YYYY HH:MM"
        return jsonify({"error": error_message_part1 + error_message_part2}), 400

    try:
        sleep_logs_collection = db.collection('SleepLogs')
        sleep_logs_collection.add({'start_time': start_time, 'end_time': end_time})
        return jsonify({"message": "Sleep duration logged successfully"}), 201
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return jsonify({"error": str(e)}), 500
from flask import request

@app.route('/api/activity/log', methods=['POST'])
def log_physical_activity():
    activity_type = request.json.get('activity_type')
    duration = request.json.get('duration')

    if not activity_type or duration is None:
        return jsonify({"error": "Activity type and duration are required"}), 400

    timestamp = datetime.now()

    try:
        physical_activities_collection = db.collection('PhysicalActivities')
        physical_activities_collection.add({'activity_type': activity_type, 'duration': duration, 'timestamp': timestamp})
        return jsonify({"message": "Activity logged successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/medication/log', methods=['POST'])
def log_medication_intake():
    medication_name = request.json.get('medication_name')
    dosage = request.json.get('dosage')
    time_taken_str = request.json.get('time_taken')
    reminder_time_str = request.json.get('reminder_time') # Get the reminder time

    if not medication_name or not dosage or not time_taken_str:
        return jsonify({"error": "Medication name, dosage, and time are required"}), 400

    try:
        time_taken = datetime.fromisoformat(time_taken_str.replace('Z', '+00:00')) # Handle potential 'Z' timezone
    except ValueError:
        return jsonify({"error": "Invalid time format for medication"}), 400

    timestamp = datetime.now()

    medication_data = {
        'medication_name': medication_name,
        'dosage': dosage,
        'time_taken': time_taken,
        'timestamp': timestamp
    }

    if reminder_time_str:
        medication_data['reminder_time'] = reminder_time_str # Add reminder time to the data

    try:
        medication_intake_collection = db.collection('MedicationIntake')
        medication_intake_collection.add(medication_data)
        return jsonify({"message": "Medication logged successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    subscription_data = request.get_json()
    # For testing, use a hardcoded user ID
    user_id = "test_user"
    try:
        # Assuming you have a Firestore collection named 'PushSubscriptions'
        push_subscriptions_ref = db.collection('PushSubscriptions')
        # Store the subscription data with the user ID as the document ID (or as a field)
        push_subscriptions_ref.document(user_id).set(subscription_data)
        print(f"Subscription stored for user: {user_id}")
        return jsonify({"success": True}), 201
    except Exception as e:
        print(f"Error storing subscription: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)