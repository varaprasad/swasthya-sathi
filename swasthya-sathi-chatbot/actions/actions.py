from typing import Any, Dict, List, Text
import re
from datetime import datetime
import os

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase using environment variable or fallback to file
if not firebase_admin._apps:
    # Use environment variable for production, file for development
    firebase_key_path = os.getenv('FIREBASE_KEY_PATH', 'swasthyasathi-firebase-firebase-adminsdk-fbsvc-f4d90b5b72.json')
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

class ActionLogWater(Action):
    def name(self) -> Text:
        return "action_log_water"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        amount = tracker.get_slot("amount")
        
        if not amount:
            dispatcher.utter_message(text="Please specify how much water you drank.")
            return []
        
        # Extract quantity in ml
        quantity_ml = self.extract_quantity_ml(amount)
        
        if quantity_ml <= 0:
            dispatcher.utter_message(text="I couldn't understand the amount. Please specify a valid quantity like '2 liters' or '500ml'.")
            return []
        
        # Store in Firebase
        try:
            db = firestore.client()
            water_data = {
                "user_id": tracker.sender_id,  # Add user identification
                "quantity": quantity_ml,
                "timestamp": datetime.now(),
                "original_input": amount  # Store original input for debugging
            }
            
            # Add to WaterIntake collection
            doc_ref = db.collection("WaterIntake").add(water_data)
            dispatcher.utter_message(text=f"Great! I've logged {amount} ({quantity_ml}ml) of water intake to your health record.")
            
        except Exception as e:
            # Log error but don't expose technical details to user
            print(f"Firebase error: {e}")
            dispatcher.utter_message(text=f"I've noted {amount} of water intake, but couldn't sync to your health record right now. Please try again later.")
            
        return []
    
    def extract_quantity_ml(self, amount_text: str) -> int:
        """Convert various water amount formats to ml"""
        if not amount_text:
            return 0
            
        amount_text = amount_text.lower().strip()
        
        # Extract numbers (handle decimals too)
        numbers = re.findall(r'\d+\.?\d*', amount_text)
        if not numbers:
            return 0
            
        try:
            quantity = float(numbers[0])
        except ValueError:
            return 0
        
        # Convert to ml based on unit
        if any(unit in amount_text for unit in ['liter', 'litre', 'l ']):
            return int(quantity * 1000)
        elif 'cup' in amount_text:
            return int(quantity * 240)
        elif 'glass' in amount_text:
            return int(quantity * 250)
        elif 'ml' in amount_text:
            return int(quantity)
        else:
            # Default assumption: if no unit specified, assume ml
            return int(quantity)