import logo from './logo.svg';
import './App.css';
import VoiceProcessor from './components/VoiceProcessor';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>
          Walkie Talkie
        </h1>
        <VoiceProcessor />
      </header>
    </div>
  );
}

export default App;
