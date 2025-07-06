import React, { useState, useEffect } from 'react';
import './HabitList.css'; // Import your HabitList CSS file

function HabitList({ language, onReminderChange }) {
  const [habits, setHabits] = useState([]);
  const [completedHabits, setCompletedHabits] = useState({});
  const backendUrl = 'http://localhost:5000'; // Your Flask backend URL

  useEffect(() => {
    console.log("HabitList useEffect is running!"); // Existing log

    const fetchHabits = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/habits`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setHabits(data);
        console.log("Habits data:", data); // ADD THIS LINE

        // Fetch completion status for today
        const habitIds = data.map(habit => habit.id);
        const today = new Date().toISOString().slice(0, 10); // Get today's date in('-');-MM-DD format

        const statusResponse = await fetch(`${backendUrl}/api/habits/status`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ habit_ids: habitIds, date: today }),
        });

        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          setCompletedHabits(statusData);
          console.log("Completion Status:", statusData);
        } else {
          console.error("Failed to fetch completion status:", statusResponse.status);
        }

      } catch (error) {
        console.error("Could not fetch habits:", error);
        // You might want to display an error message to the user here
      }
    };

    fetchHabits();
  }, [backendUrl]); // Re-fetch if backendUrl changes

  const handleHabitCompletionToggle = (habitId, isChecked) => {
    setCompletedHabits(prevCompleted => ({
      ...prevCompleted,
      [habitId]: isChecked,
    }));
    console.log(`Habit ${habitId} is now ${isChecked ? 'completed' : 'not completed'}`);

    fetch(`${backendUrl}/api/habits/complete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ habit_id: habitId }),
    })
      .then(response => {
        if (response.ok) {
          console.log(`Habit ${habitId} completion recorded on backend`);
        } else {
          console.error(`Failed to record habit ${habitId} completion on backend`);
        }
        return response.json(); // Optionally log the json response
      })
      .then(data => {
        console.log("Completion API Response:", data);
      })
      .catch(error => {
        console.error("Error recording habit completion:", error);
      });
  };

  return (
    <div className="habit-list-section">
      <h1>Habit List</h1>
      <ul className="habit-list-ul">
        {habits.map(habit => (
          <li key={habit.id} className="habit-list-item">
            <input
              type="checkbox"
              style={{ accentColor: '#3498DB' }}
              checked={completedHabits[habit.id] || false}
              onChange={(event) => handleHabitCompletionToggle(habit.id, event.target.checked)}
            />
            <span className="habit-text">{language === 'te' ? habit.name_te : habit.name_en}</span>
            <div className="habit-reminder-group">
              <label htmlFor={`habitReminder-${habit.id}`}>Set Reminder:</label>
              <input
                type="time"
                id={`habitReminder-${habit.id}`}
                onChange={(event) => onReminderChange(habit.id, event.target.value)}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default HabitList;