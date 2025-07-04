from flask import Flask, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Firebase Admin SDK (assuming you've set up the credentials)
#cred = credentials.Certificate('path/to/your/serviceAccountKey.json') # Replace with your key file path
cred = credentials.Certificate('credentials/swasthyasathi-firebase-firebase-adminsdk-fbsvc-f4d90b5b72.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

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

if __name__ == '__main__':
    app.run(debug=True)