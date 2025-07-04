import React, { useState } from 'react'; // Import useState
import HabitList from './HabitList';
import logo from './logo.svg';
import './App.css';

function App() {
  const [language, setLanguage] = useState('en'); // State for current language ('en' or 'te')

  const toggleLanguage = () => {
    setLanguage(prevLang => (prevLang === 'en' ? 'te' : 'en'));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>SwasthyaSathi</h1>
        <button onClick={toggleLanguage}>
          {language === 'en' ? 'Switch to తెలుగు' : 'Switch to English'}
        </button>
      </header>
      <main>
        <HabitList language={language} /> {/* Pass the language state as a prop */}
      </main>
    </div>
  );
}

export default App;