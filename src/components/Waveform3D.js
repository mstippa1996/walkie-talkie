import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const GREEN = '#33ff66';
const BG = '#04120a';

const POINTS = 128; // samples per waveform line
const LINES = 70; // how many past waveforms stay visible
const WIDTH = 8; // x extent of each line
const DEPTH = 9; // how far the oldest line recedes
const AMPLITUDE = 1.6; // vertical scale of the wave
const FRAME_SKIP = 2; // capture a new line every N frames (controls scroll speed)
const TARGET = new THREE.Vector3(0, 0, -DEPTH / 2);

function Controls() {
  const { camera, gl } = useThree();
  const controlsRef = useRef(null);

  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.target.copy(TARGET);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2; // keep the camera above the floor
    controls.minDistance = 3;
    controls.maxDistance = 20;
    controls.update();
    controlsRef.current = controls;
    return () => controls.dispose();
  }, [camera, gl]);

  useFrame(() => controlsRef.current && controlsRef.current.update());
  return null;
}

function WaveTerrain({ analyser }) {
  const frameCount = useRef(0);

  // One THREE.Line per history slot. Slot 0 = newest (z = 0), last = oldest.
  const lines = useMemo(() => {
    const list = [];
    for (let k = 0; k < LINES; k++) {
      const positions = new Float32Array(POINTS * 3);
      const z = -(k / (LINES - 1)) * DEPTH;
      for (let i = 0; i < POINTS; i++) {
        positions[i * 3] = (i / (POINTS - 1) - 0.5) * WIDTH;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = z;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({
        color: GREEN,
        transparent: true,
        opacity: Math.pow(1 - k / LINES, 1.5), // older lines fade out
      });
      list.push(new THREE.Line(geometry, material));
    }
    return list;
  }, []);

  useEffect(
    () => () =>
      lines.forEach((line) => {
        line.geometry.dispose();
        line.material.dispose();
      }),
    [lines]
  );

  const samples = useMemo(() => (analyser ? new Uint8Array(analyser.fftSize) : null), [analyser]);

  useFrame(() => {
    frameCount.current += 1;
    if (frameCount.current % FRAME_SKIP !== 0) return;

    // Scroll history back: each line takes the y values of the one in front of it
    for (let k = LINES - 1; k > 0; k--) {
      const dst = lines[k].geometry.attributes.position;
      const src = lines[k - 1].geometry.attributes.position;
      for (let i = 0; i < POINTS; i++) dst.array[i * 3 + 1] = src.array[i * 3 + 1];
      dst.needsUpdate = true;
    }

    // Newest line comes from the mic (flat when there is no analyser)
    const head = lines[0].geometry.attributes.position;
    if (analyser && samples) {
      analyser.getByteTimeDomainData(samples); // 0-255, 128 = silence
      const stride = samples.length / POINTS;
      for (let i = 0; i < POINTS; i++) {
        head.array[i * 3 + 1] = ((samples[Math.floor(i * stride)] - 128) / 128) * AMPLITUDE;
      }
    } else {
      for (let i = 0; i < POINTS; i++) head.array[i * 3 + 1] = 0;
    }
    head.needsUpdate = true;
  });

  return (
    <group>
      {lines.map((line, k) => (
        <primitive key={k} object={line} />
      ))}
    </group>
  );
}

function Floor() {
  const grid = useMemo(() => {
    const helper = new THREE.GridHelper(WIDTH * 1.4, 20, GREEN, GREEN);
    helper.material.transparent = true;
    helper.material.opacity = 0.12;
    helper.position.set(0, -AMPLITUDE - 0.1, TARGET.z);
    return helper;
  }, []);

  useEffect(
    () => () => {
      grid.geometry.dispose();
      grid.material.dispose();
    },
    [grid]
  );

  return <primitive object={grid} />;
}

export default function Waveform3D({ analyser, width = 480, height = 320 }) {
  return (
    <div
      style={{
        display: 'inline-block',
        padding: '12px',
        background: '#2b2b2b',
        border: '3px solid #111',
        borderRadius: '12px',
        boxShadow: 'inset 0 0 0 2px #444, 0 4px 12px rgba(0,0,0,0.5)',
      }}
    >
      <div
        style={{
          width,
          height,
          borderRadius: '6px',
          border: '2px solid #000',
          overflow: 'hidden',
          boxShadow: 'inset 0 0 20px rgba(0,0,0,0.8), 0 0 12px rgba(51,255,102,0.25)',
        }}
      >
        <Canvas camera={{ position: [0, 2.6, 4.5], fov: 50 }}>
          <color attach="background" args={[BG]} />
          <fog attach="fog" args={[BG, 6, 16]} />
          <Floor />
          <WaveTerrain analyser={analyser} />
          <Controls />
        </Canvas>
      </div>
    </div>
  );
}
