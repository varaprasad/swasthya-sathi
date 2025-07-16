import React, { useState, useEffect, useCallback } from 'react';
import './HabitList.css'; // Import your HabitList CSS file

function HabitList({ language, onReminderChange, onCompleteHabit }) {
  const [habits, setHabits] = useState([]);
  const [completedHabits, setCompletedHabits] = useState({});
  const backendUrl = 'http://localhost:5000'; // Your Flask backend URL
  const [isSyncing, setIsSyncing] = useState(false);

  const fetchHabitsData = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/habits`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHabits(data);
    } catch (error) {
      console.error("Could not fetch habits:", error);
    }
  }, [backendUrl]);

  const fetchCompletionStatus = useCallback(async () => {
    const habitIds = habits.map(habit => habit.id);
    const today = new Date().toISOString().slice(0, 10); // Get today's date in YYYY-MM-DD format

    try {
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
      } else {
        console.error("Failed to fetch completion status:", statusResponse.status);
      }
    } catch (error) {
      console.error("Failed to fetch completion status:", error);
    } finally {
      setIsSyncing(false);
    }
  }, [backendUrl, habits]);

  useEffect(() => {
    console.log("HabitList useEffect is running!");
    fetchHabitsData();
  }, [fetchHabitsData]); // Re-fetch if backendUrl changes

  useEffect(() => {
    if (habits.length > 0) {
      fetchCompletionStatus();
    }
  }, [habits, fetchCompletionStatus]); // Fetch completion status when habits are loaded

  const handleHabitCompletionToggle = async (habitId, isChecked) => {
    setIsSyncing(true);
    // Optimistic update
    setCompletedHabits(prevCompleted => ({
      ...prevCompleted,
      [habitId]: isChecked,
    }));
    console.log(`Habit ${habitId} is now ${isChecked ? 'completed' : 'not completed'}`);
    await onCompleteHabit(habitId, isChecked);
    // Refetch completion status after attempting to update the backend
    fetchCompletionStatus();
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