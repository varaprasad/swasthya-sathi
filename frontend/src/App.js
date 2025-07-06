import React, { useState, useEffect } from 'react';
import HabitList from './HabitList';
import './App.css'; // Import your CSS file

function App() {
  const [language, setLanguage] = useState('en');
  const [waterQuantity, setWaterQuantity] = useState('');
  const [waterLogMessage, setWaterLogMessage] = useState('');
  const [todaysWaterIntake, setTodaysWaterIntake] = useState(0);
  const [sleepStartTime, setSleepStartTime] = useState('');
  const [sleepEndTime, setSleepEndTime] = useState('');
  const [sleepLogMessage, setSleepLogMessage] = useState('');
  const [activityType, setActivityType] = useState('');
  const [activityDuration, setActivityDuration] = useState('');
  const [activityLogMessage, setActivityLogMessage] = useState('');
  const [medicationName, setMedicationName] = useState('');
  const [medicationDosage, setMedicationDosage] = useState('');
  const [medicationTime, setMedicationTime] = useState('');
  const [medicationReminderTime, setMedicationReminderTime] = useState('');
  const [medicationLogMessage, setMedicationLogMessage] = useState('');
  const [waterReminderTime, setWaterReminderTime] = useState('');
  const [habitReminderTimes, setHabitReminderTimes] = useState({});
  const backendUrl = 'http://localhost:5000';

  const fetchTodaysWaterIntake = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/water/today`);
      if (response.ok) {
        const data = await response.json();
        setTodaysWaterIntake(data.total_intake_ml);
      } else {
        console.error("Failed to fetch today's water intake:", response.status);
      }
    } catch (error) {
      console.error("Error fetching today's water intake:", error);
    }
  };

  useEffect(() => {
    fetchTodaysWaterIntake();
    // You might want to fetch this data periodically or when a new log is added
  }, [backendUrl]);

  useEffect(() => {
    if (Notification.permission === 'granted') {
      new Notification('Test Notification from useEffect', { body: 'This is a test notification triggered on page load (if permission is granted)!' });
    }
  }, []);

  useEffect(() => {
    console.log('Habit Reminder useEffect triggered!');
    if (habitReminderTimes && Object.keys(habitReminderTimes).length > 0) {
      Notification.requestPermission().then(permission => {
        console.log('Notification permission in Habit Reminder useEffect:', permission);
        if (permission === 'granted') {
          console.log('Habit Reminder useEffect: Permission granted. Fetching habits...');
          // Fetch habits to get the names based on IDs
          const fetchHabits = async () => {
            try {
              console.log('Habit Reminder useEffect: Inside fetchHabits');
              const response = await fetch(`${backendUrl}/api/habits`);
              if (response.ok) {
                const habitsData = await response.json();
                console.log('Habit Reminder useEffect: Habits data received:', habitsData);
                for (const habitId in habitReminderTimes) {
                  console.log('Habit Reminder useEffect: habitId (type):', habitId, typeof habitId);
                  const reminderTime = habitReminderTimes[habitId];
                  if (reminderTime) {
                    const habit = habitsData.find(h => {
                      console.log('Habit Reminder useEffect: habit.id (type):', h.id, typeof h.id);
                      return h.id === habitId; // Compare as strings directly
                    });
                    console.log('Habit Reminder useEffect: Found habit:', habit);
                    if (habit) {
                      const now = new Date();
                      const reminderDate = new Date();
                      const [hours, minutes] = reminderTime.split(':');
                      reminderDate.setHours(parseInt(hours));
                      reminderDate.setMinutes(parseInt(minutes));
                      reminderDate.setSeconds(0);
                      reminderDate.setMilliseconds(0);

                      console.log('Habit Reminder useEffect: Reminder Date (calculated):', reminderDate);

                      let delay = reminderDate.getTime() - now.getTime();
                      console.log('Habit Reminder useEffect: Calculated Delay:', delay);

                      if (delay <= 0) {
                        reminderDate.setDate(reminderDate.getDate() + 1);
                        delay = reminderDate.getTime() - now.getTime();
                        console.log('Habit Reminder useEffect: Delay adjusted for next day:', delay);
                      }

                      console.log('Habit Reminder useEffect: Scheduling notification with delay:', delay);
                      setTimeout(() => {
                        new Notification('Habit Reminder', {
                          body: `Time to do your habit: ${habit.name_en}`, // Assuming English for now
                        });
                      }, delay);
                      console.log(`Habit Reminder useEffect: Reminder scheduled for habit ${habit.name_en} at ${reminderTime} with delay: ${delay}`);
                    }
                  }
                }
              } else {
                console.error('Habit Reminder useEffect: Failed to fetch habits.');
              }
            } catch (error) {
              console.error("Habit Reminder useEffect: Error fetching habits:", error);
            }
          };
          fetchHabits();
        }
      });
    }
  }, [habitReminderTimes, backendUrl]);

  const handleLanguageToggle = () => {
    setLanguage(prevLanguage => (prevLanguage === 'en' ? 'te' : 'en'));
  };

  const handleWaterInputChange = (event) => {
    setWaterQuantity(event.target.value);
  };

  const handleLogWater = async () => {
    console.log('Water Reminder Time:', waterReminderTime);
    if (!waterQuantity) {
      setWaterLogMessage('Please enter the quantity of water.');
      return;
    }

    try {
      const response = await fetch(`${backendUrl}/api/water/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ quantity: parseInt(waterQuantity) }),
      });

      const data = await response.json();
      setWaterLogMessage(data.message || (response.ok ? 'Water intake logged successfully.' : 'Failed to log water intake.'));
      setWaterQuantity(''); // Clear the input field
      // After logging, re-fetch today's total
      fetchTodaysWaterIntake();

      // Schedule reminder if reminder time is set
      if (waterReminderTime) {
        Notification.requestPermission().then(permission => {
          if (permission === 'granted') {
            const now = new Date();
            const reminderDate = new Date(); // Using current date for water reminder
            const [hours, minutes] = waterReminderTime.split(':');
            reminderDate.setHours(parseInt(hours));
            reminderDate.setMinutes(parseInt(minutes));
            reminderDate.setSeconds(0);
            reminderDate.setMilliseconds(0);

            let delay = reminderDate.getTime() - now.getTime();

            if (delay <= 0) {
              // If reminder time is in the past, schedule for the next day
              reminderDate.setDate(reminderDate.getDate() + 1);
              delay = reminderDate.getTime() - now.getTime();
            }

            setTimeout(() => {
              new Notification('Water Reminder', {
                body: 'Time to drink water!',
              });
            }, delay);

            setWaterReminderTime(''); // Clear reminder time after setting
          } else {
            setWaterLogMessage('Notification permission was not granted.');
          }
        });
      }

    } catch (error) {
      console.error("Error logging water intake:", error);
      setWaterLogMessage('Failed to log water intake.');
    }
  };

  const handleLogSleep = async () => {
    if (!sleepStartTime || !sleepEndTime) {
      setSleepLogMessage('Please select both start and end times.');
      return;
    }

    try {
      const response = await fetch(`${backendUrl}/api/sleep/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ start_time: sleepStartTime, end_time: sleepEndTime }),
      });

      const data = await response.json();
      setSleepLogMessage(data.message || (response.ok ? 'Sleep duration logged successfully.' : 'Failed to log sleep.'));
      setSleepStartTime(''); // Clear the input fields
      setSleepEndTime('');
    } catch (error) {
      console.error("Error logging sleep:", error);
      setSleepLogMessage('Failed to log sleep.');
    }
  };

  const handleLogActivity = async () => {
    if (!activityType || !activityDuration) {
      setActivityLogMessage('Please enter both activity type and duration.');
      return;
    }

    try {
      const response = await fetch(`${backendUrl}/api/activity/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          activity_type: activityType,
          duration: parseInt(activityDuration),
        }),
      });

      const data = await response.json();
      setActivityLogMessage(data.message || (response.ok ? 'Activity logged successfully.' : 'Failed to log activity.'));
      setActivityType('');
      setActivityDuration('');
    } catch (error) {
      console.error("Error logging activity:", error);
      setActivityLogMessage('Failed to log activity.');
    }
  };

  const handleLogMedication = async () => {
    console.log('Medication Reminder Time:', medicationReminderTime);
    if (!medicationName || !medicationDosage || !medicationTime) {
      setMedicationLogMessage('Please enter medication name, dosage, and time.');
      return;
    }

    try {
      const response = await fetch(`${backendUrl}/api/medication/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          medication_name: medicationName,
          dosage: medicationDosage,
          time_taken: medicationTime,
          reminder_time: medicationReminderTime, // Include reminder time here
        }),
      });

      const data = await response.json();
      setMedicationLogMessage(data.message || (response.ok ? 'Medication logged successfully.' : 'Failed to log medication.'));
      setMedicationName('');
      setMedicationDosage('');
      setMedicationTime('');
      setMedicationReminderTime(''); // Clear reminder time after sending
    } catch (error) {
      console.error("Error logging medication:", error);
      setMedicationLogMessage('Failed to log medication.');
    }
  };

  const handleHabitReminderChange = (habitId, reminderTime) => {
    setHabitReminderTimes(prevTimes => ({
      ...prevTimes,
      [habitId]: reminderTime,
    }));
    console.log(`Reminder set for habit ${habitId} at:`, reminderTime);
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <h1 className="app-title">SwasthyaSathi</h1>
        <button onClick={handleLanguageToggle} className="language-toggle-button">
          Switch to {language === 'en' ? 'తెలుగు' : 'English'}
        </button>
      </div>

      <div className="section-card">
        <HabitList language={language} onReminderChange={handleHabitReminderChange} />
      </div>

      {/* Water Intake Tracking Section */}
      <div className="section-card">
        <h2>Track Water Intake</h2>
        <div className="form-group horizontal">
          <label htmlFor="waterQuantity">Enter quantity (ml):</label>
          <input
            type="number"
            id="waterQuantity"
            placeholder="e.g., 500"
            value={waterQuantity}
            onChange={handleWaterInputChange}
          />
        </div>
        <button onClick={handleLogWater} className="primary-button">Log Water</button>
        {waterLogMessage && <p className="log-message">{waterLogMessage}</p>}

        <div className="form-group horizontal" style={{ marginTop: '20px' }}>
          <label htmlFor="waterReminder">Set Reminder:</label>
          <input
            type="time"
            id="waterReminder"
            onChange={(e) => setWaterReminderTime(e.target.value)}
            value={waterReminderTime}
          />
        </div>
      </div>

      {/* Display Today's Water Intake */}
      <div className="section-card">
        <h2>Today's Total Water Intake</h2>
        <p style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#3498DB' }}>{todaysWaterIntake} ml</p>
      </div>

      {/* Sleep Duration Tracking Section */}
      <div className="section-card">
        <h2>Track Sleep Duration</h2>
        <div className="form-group horizontal">
          <label htmlFor="sleepStart">Sleep Start Time:</label>
          <input
            type="datetime-local"
            id="sleepStart"
            onChange={(e) => setSleepStartTime(e.target.value)}
            value={sleepStartTime}
          />
        </div>
        <div className="form-group horizontal">
          <label htmlFor="sleepEnd">Sleep End Time:</label>
          <input
            type="datetime-local"
            id="sleepEnd"
            onChange={(e) => setSleepEndTime(e.target.value)}
            value={sleepEndTime}
          />
        </div>
        <button onClick={handleLogSleep} className="primary-button">Log Sleep</button>
        {sleepLogMessage && <p className="log-message">{sleepLogMessage}</p>}
      </div>

      {/* Physical Activity Tracking Section */}
      <div className="section-card">
        <h2>Track Physical Activity</h2>
        <div className="form-group horizontal">
          <label htmlFor="activityType">Activity Type:</label>
          <input
            type="text"
            id="activityType"
            placeholder="e.g., Running, Yoga"
            onChange={(e) => setActivityType(e.target.value)}
            value={activityType}
          />
        </div>
        <div className="form-group horizontal">
          <label htmlFor="activityDuration">Duration (in minutes):</label>
          <input
            type="number"
            id="activityDuration"
            placeholder="e.g., 30"
            onChange={(e) => setActivityDuration(e.target.value)}
            value={activityDuration}
          />
        </div>
        <button onClick={handleLogActivity} className="primary-button">Log Activity</button>
        {activityLogMessage && <p className="log-message">{activityLogMessage}</p>}
      </div>

      {/* Medication Intake Tracking Section */}
      <div className="section-card">
        <h2>Track Medication Intake</h2>
        <div className="form-group horizontal">
          <label htmlFor="medicationName">Medication Name:</label>
          <input
            type="text"
            id="medicationName"
            placeholder="e.g., Paracetamol"
            onChange={(e) => setMedicationName(e.target.value)}
            value={medicationName}
          />
        </div>
        <div className="form-group horizontal">
          <label htmlFor="medicationDosage">Dosage:</label>
          <input
            type="text"
            id="medicationDosage"
            placeholder="e.g., 500mg, 1 tablet"
            onChange={(e) => setMedicationDosage(e.target.value)}
            value={medicationDosage}
          />
        </div>
        <div className="form-group horizontal">
          <label htmlFor="medicationTime">Time Taken:</label>
          <input
            type="datetime-local"
            id="medicationTime"
            onChange={(e) => setMedicationTime(e.target.value)}
            value={medicationTime}
          />
        </div>
        <div className="form-group horizontal">
          <label htmlFor="medicationReminder">Set Reminder:</label>
          <input
            type="time"
            id="medicationReminder"
            onChange={(e) => setMedicationReminderTime(e.target.value)}
            value={medicationReminderTime}
          />
        </div>
        <button onClick={handleLogMedication} className="primary-button">Log Medication</button>
        {medicationLogMessage && <p className="log-message">{medicationLogMessage}</p>}
      </div>
    </div>
  );
}

export default App;