import React, { useState, useEffect } from 'react';
import HabitList from './HabitList';

function App() {
  const [language, setLanguage] = useState('en');
  const [waterQuantity, setWaterQuantity] = useState('');
  const [waterLogMessage, setWaterLogMessage] = useState('');
  const [todaysWaterIntake, setTodaysWaterIntake] = useState(0); // New state for total intake
  const [sleepStartTime, setSleepStartTime] = useState(''); // New state for sleep start time
  const [sleepEndTime, setSleepEndTime] = useState('');     // New state for sleep end time
  const [sleepLogMessage, setSleepLogMessage] = useState(''); // New state for sleep log message
  const [activityType, setActivityType] = useState('');       // New state for activity type
  const [activityDuration, setActivityDuration] = useState(''); // New state for activity duration
  const [activityLogMessage, setActivityLogMessage] = useState(''); // New state for activity log message
  const [medicationName, setMedicationName] = useState('');     // New state for medication name
  const [medicationDosage, setMedicationDosage] = useState(''); // New state for medication dosage
  const [medicationTime, setMedicationTime] = useState('');     // New state for medication time
  const [medicationReminderTime, setMedicationReminderTime] = useState(''); // New state for medication reminder time
  const [medicationLogMessage, setMedicationLogMessage] = useState(''); // New state for medication log message
  const [waterReminderTime, setWaterReminderTime] = useState(''); // New state for water reminder time
  const [habitReminderTimes, setHabitReminderTimes] = useState({}); // State to store reminder times for each habit
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
  }, [backendUrl]); // Fetch again if backendUrl changes

  useEffect(() => {
    if (Notification.permission === 'granted') {
      new Notification('Test Notification from useEffect', { body: 'This is a test notification triggered on page load (if permission is granted)!' });
    }
  }, []);

  useEffect(() => {
    console.log('Habit Reminder useEffect triggered!');
    if (habitReminderTimes && Object.keys(habitReminderTimes).length > 0) {
      Notification.requestPermission().then(permission => {
        console.log('Notification permission in Habit Reminder useEffect:', permission); // Added log
        if (permission === 'granted') {
          console.log('Habit Reminder useEffect: Permission granted. Fetching habits...'); // Added log
          // Fetch habits to get the names based on IDs
          const fetchHabits = async () => {
            try {
              console.log('Habit Reminder useEffect: Inside fetchHabits'); // Added log
              const response = await fetch(`${backendUrl}/api/habits`);
              if (response.ok) {
                const habitsData = await response.json();
                console.log('Habit Reminder useEffect: Habits data received:', habitsData); // Added log
                for (const habitId in habitReminderTimes) {
                  console.log('Habit Reminder useEffect: habitId (type):', habitId, typeof habitId); // ADD THIS LINE
                  const reminderTime = habitReminderTimes[habitId];
                  if (reminderTime) {
                    const habit = habitsData.find(h => {
                      console.log('Habit Reminder useEffect: habit.id (type):', h.id, typeof h.id); // ADD THIS LINE
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

                      console.log('Habit Reminder useEffect: Scheduling notification with delay:', delay); // Added log
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
                console.error('Habit Reminder useEffect: Failed to fetch habits.'); // Added log
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
    console.log('Water Reminder Time:', waterReminderTime); // Optional: Add this line
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
    console.log('Medication Reminder Time:', medicationReminderTime); // Optional line
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
    console.log(`Reminder set for habit ${habitId} at:`, reminderTime); // Optional log
  };

  return (
    <div>
      <h1>SwasthyaSathi</h1>
      <button onClick={handleLanguageToggle}>
        Switch to {language === 'en' ? 'తెలుగు' : 'English'}
      </button>
      <HabitList language={language} onReminderChange={handleHabitReminderChange} />

      {/* Water Intake Tracking Section */}
      <h2>Track Water Intake</h2>
      <div>
        <input
          type="number"
          placeholder="Enter quantity (ml)"
          value={waterQuantity}
          onChange={handleWaterInputChange}
        />
        <button onClick={handleLogWater}>Log Water</button>
        {waterLogMessage && <p>{waterLogMessage}</p>}
      </div>
      <div>
        <label htmlFor="waterReminder">Set Reminder:</label>
        <input
          type="time"
          id="waterReminder"
          onChange={(e) => setWaterReminderTime(e.target.value)}
          value={waterReminderTime}
        />
      </div>

      {/* Display Today's Water Intake */}
      <h2>Today's Total Water Intake</h2>
      <p>{todaysWaterIntake} ml</p>

      {/* Sleep Duration Tracking Section */}
      <h2>Track Sleep Duration</h2>
      <div>
        <label htmlFor="sleepStart">Sleep Start Time:</label>
        <input
          type="datetime-local"
          id="sleepStart"
          onChange={(e) => setSleepStartTime(e.target.value)}
          value={sleepStartTime}
        />
      </div>
      <div>
        <label htmlFor="sleepEnd">Sleep End Time:</label>
        <input
          type="datetime-local"
          id="sleepEnd"
          onChange={(e) => setSleepEndTime(e.target.value)}
          value={sleepEndTime}
        />
      </div>
      <button onClick={handleLogSleep}>Log Sleep</button>
      {sleepLogMessage && <p>{sleepLogMessage}</p>}

      {/* Physical Activity Tracking Section */}
      <h2>Track Physical Activity</h2>
      <div>
        <label htmlFor="activityType">Activity Type:</label>
        <input
          type="text"
          id="activityType"
          onChange={(e) => setActivityType(e.target.value)}
          value={activityType}
        />
      </div>
      <div>
        <label htmlFor="activityDuration">Duration (in minutes):</label>
        <input
          type="number"
          id="activityDuration"
          onChange={(e) => setActivityDuration(e.target.value)}
          value={activityDuration}
        />
      </div>
      <button onClick={handleLogActivity}>Log Activity</button>
      {activityLogMessage && <p>{activityLogMessage}</p>}

      {/* Medication Intake Tracking Section */}
      <h2>Track Medication Intake</h2>
      <div>
        <label htmlFor="medicationName">Medication Name:</label>
        <input
          type="text"
          id="medicationName"
          onChange={(e) => setMedicationName(e.target.value)}
          value={medicationName}
        />
      </div>
      <div>
        <label htmlFor="medicationDosage">Dosage:</label>
        <input
          type="text"
          id="medicationDosage"
          onChange={(e) => setMedicationDosage(e.target.value)}
          value={medicationDosage}
        />
      </div>
      <div>
        <label htmlFor="medicationTime">Time Taken:</label>
        <input
          type="datetime-local"
          id="medicationTime"
          onChange={(e) => setMedicationTime(e.target.value)}
          value={medicationTime}
        />
      </div>
      <div>
        <label htmlFor="medicationReminder">Set Reminder:</label>
        <input
          type="time"
          id="medicationReminder"
          onChange={(e) => setMedicationReminderTime(e.target.value)}
          value={medicationReminderTime}
        />
      </div>
      <button onClick={handleLogMedication}>Log Medication</button>
      {medicationLogMessage && <p>{medicationLogMessage}</p>}
    </div>
  );
}

export default App;