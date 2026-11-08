import { useEffect, useRef, useState } from 'react';

// YAMNet (AudioSet) class indices, from yamnet_class_map.csv
const SPEECH_CLASSES = [0, 1, 2, 3, 4]; // Speech, Child speech, Conversation, Narration, Babbling
const SINGING_CLASSES = [24, 25, 29, 30]; // Singing, Choir, Child singing, Synthetic singing

const MODEL_SAMPLE_RATE = 16000; // YAMNet expects mono 16 kHz in [-1, 1]
const WINDOW_SECONDS = 1.5; // audio fed to the model each time (model needs >= 0.975 s)
const INFERENCE_INTERVAL_MS = 500;
const SILENCE_RMS = 0.01; // below this we call it quiet and skip the model
const SMOOTHING = 0.5; // EMA weight of the newest result (higher = more responsive)
const SINGING_THRESHOLD = 0.3; // smoothed singing score needed to call it singing

// Self-hosted copy first (put the model files in public/models/yamnet/), then Kaggle.
const LOCAL_MODEL_URL = process.env.REACT_APP_YAMNET_URL || '/models/yamnet/model.json';
const KAGGLE_MODEL_URL = 'https://www.kaggle.com/models/google/yamnet/tfJs/tfjs/1';

// AudioWorklet that forwards raw mic samples to the main thread in chunks.
const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(4096);
    this.filled = 0;
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        this.chunk[this.filled++] = channel[i];
        if (this.filled === this.chunk.length) {
          this.port.postMessage(this.chunk.slice());
          this.filled = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

async function loadModel(tf) {
  try {
    return await tf.loadGraphModel(LOCAL_MODEL_URL);
  } catch (localError) {
    console.warn('Local YAMNet model not found, trying Kaggle:', localError.message);
    return tf.loadGraphModel(KAGGLE_MODEL_URL, { fromTFHub: true });
  }
}

// Linear resample to 16 kHz
function resample(input, fromRate) {
  if (fromRate === MODEL_SAMPLE_RATE) return input;
  const ratio = fromRate / MODEL_SAMPLE_RATE;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

/**
 * Classifies the mic audio as singing / speaking / quiet using YAMNet (TensorFlow.js).
 * Pass the AnalyserNode from VoiceProcessor (or null when the mic is off).
 */
export default function useSingingDetector(analyser) {
  const [state, setState] = useState({
    status: 'idle', // idle | loading | ready | error
    label: 'quiet', // quiet | speaking | singing
    singingScore: 0,
    speechScore: 0,
    error: null,
  });
  const smoothedRef = useRef({ singing: 0, speech: 0 });

  useEffect(() => {
    if (!analyser) {
      setState((s) => ({ ...s, status: 'idle', label: 'quiet', singingScore: 0, speechScore: 0 }));
      return undefined;
    }

    let cancelled = false;
    let intervalId;
    let workletNode;
    let silentGain;
    let blobUrl;
    smoothedRef.current = { singing: 0, speech: 0 };

    const context = analyser.context;
    const ringSize = Math.ceil(context.sampleRate * WINDOW_SECONDS);
    const ring = new Float32Array(ringSize);
    let written = 0; // total samples received

    const start = async () => {
      setState((s) => ({ ...s, status: 'loading', error: null }));
      try {
        const tf = await import('@tensorflow/tfjs'); // lazy-load so it stays out of the main bundle
        await tf.ready();
        const model = await loadModel(tf);
        if (cancelled) return;

        // Capture mic samples by tapping the analyser (it passes audio through unchanged)
        blobUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
        await context.audioWorklet.addModule(blobUrl);
        if (cancelled) return;

        workletNode = new AudioWorkletNode(context, 'capture-processor');
        workletNode.port.onmessage = (event) => {
          const chunk = event.data;
          for (let i = 0; i < chunk.length; i++) {
            ring[written % ringSize] = chunk[i];
            written++;
          }
        };
        silentGain = context.createGain();
        silentGain.gain.value = 0; // keep the worklet running without playing the mic back
        analyser.connect(workletNode);
        workletNode.connect(silentGain);
        silentGain.connect(context.destination);

        setState((s) => ({ ...s, status: 'ready' }));

        let busy = false;
        intervalId = setInterval(async () => {
          if (busy || written < ringSize) return; // wait until the window is full
          busy = true;
          try {
            // Unroll the ring buffer into time order
            const samples = new Float32Array(ringSize);
            const start = written % ringSize;
            samples.set(ring.subarray(start), 0);
            samples.set(ring.subarray(0, start), ringSize - start);

            let sumSquares = 0;
            for (let i = 0; i < samples.length; i++) sumSquares += samples[i] * samples[i];
            const rms = Math.sqrt(sumSquares / samples.length);

            let singing = 0;
            let speech = 0;
            if (rms >= SILENCE_RMS) {
              const waveform = resample(samples, context.sampleRate);
              const scores = tf.tidy(() => {
                const outputs = model.predict(tf.tensor1d(waveform));
                const frameScores = Array.isArray(outputs) ? outputs[0] : outputs;
                return frameScores.mean(0); // average over the 0.96 s frames
              });
              const values = await scores.data();
              scores.dispose();
              singing = SINGING_CLASSES.reduce((sum, c) => sum + values[c], 0);
              speech = SPEECH_CLASSES.reduce((sum, c) => sum + values[c], 0);
            }
            if (cancelled) return;

            const prev = smoothedRef.current;
            const next = {
              singing: prev.singing * (1 - SMOOTHING) + singing * SMOOTHING,
              speech: prev.speech * (1 - SMOOTHING) + speech * SMOOTHING,
            };
            smoothedRef.current = next;

            let label = 'quiet';
            if (rms >= SILENCE_RMS) {
              label = next.singing >= SINGING_THRESHOLD && next.singing > next.speech ? 'singing' : 'speaking';
            }
            setState((s) => ({ ...s, label, singingScore: next.singing, speechScore: next.speech }));
          } catch (error) {
            console.error('Singing detection failed:', error);
          } finally {
            busy = false;
          }
        }, INFERENCE_INTERVAL_MS);
      } catch (error) {
        console.error('Could not start singing detector:', error);
        if (!cancelled) setState((s) => ({ ...s, status: 'error', error: error.message }));
      }
    };

    start();

    return () => {
      cancelled = true;
      clearInterval(intervalId);
      try {
        if (workletNode) {
          analyser.disconnect(workletNode);
          workletNode.disconnect();
          workletNode.port.onmessage = null;
        }
        if (silentGain) silentGain.disconnect();
      } catch (e) {
        // nodes may already be torn down with the audio context
      }
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [analyser]);

  return { ...state, isSinging: state.label === 'singing' };
}
