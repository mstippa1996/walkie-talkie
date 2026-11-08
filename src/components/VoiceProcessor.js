import React, { useState, useRef } from 'react';
import Oscilloscope from './Oscilloscope';
import Waveform3D from './Waveform3D';

export default function VoiceProcessor() {
  const [isListening, setIsListening] = useState(false);
  const [analyser, setAnalyser] = useState(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  const startListening = async () => {
    try {
      // 1. Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // 2. Create AudioContext
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContext();
      audioContextRef.current = context;

      // 3. Create a source from the microphone stream
      const source = context.createMediaStreamSource(stream);
      sourceRef.current = source;

      // 4. Create an AnalyserNode (useful for volume levels or visualizers)
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;
      setAnalyser(analyser);

      setIsListening(true);
      console.log('Audio Context is running and connected to microphone.');
    } catch (error) {
      console.error('Error accessing the microphone:', error);
    }
  };

  const stopListening = () => {
    cancelAnimationFrame(rafRef.current);

    // Stop all media tracks (turns off mic recording indicator)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    // Close the audio context to free up memory
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }

    setAnalyser(null);
    setIsListening(false);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h3>React Web Audio API Voice Processor</h3>
      <p>Status: {isListening ? '🎤 Listening & Processing' : '⏹️ Stopped'}</p>

      <div style={{ marginBottom: '16px' }}>
        <Oscilloscope analyser={analyser} />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <Waveform3D analyser={analyser} />
      </div>

      {!isListening ? (
        <button onClick={startListening} style={{ padding: '10px 20px' }}>
          Start Microphone
        </button>
      ) : (
        <button onClick={stopListening} style={{ padding: '10px 20px', background: '#ff4d4d', color: '#fff' }}>
          Stop Microphone
        </button>
      )}
    </div>
  );
}