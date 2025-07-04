import React, { useState, useEffect } from 'react';

function HabitList({ language }) {
  const [habits, setHabits] = useState([]);
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
      } catch (error) {
        console.error("Could not fetch habits:", error);
        // You might want to display an error message to the user here
      }
    };

    fetchHabits();
  }, [backendUrl]); // Re-fetch if backendUrl changes

  return (
    <div>
      <h1>Habit List</h1>
      <ul>
        {habits.map(habit => (
          <li key={habit.id}>
            {language === 'te' ? habit.name_te : habit.name_en}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default HabitList;