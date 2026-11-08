import React, { useEffect, useRef } from 'react';

const GREEN = '#33ff66';
const BG = '#04120a';
const GRID = 'rgba(51, 255, 102, 0.15)';

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= width; x += width / 10) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = 0; y <= height; y += height / 6) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();
}

export default function Oscilloscope({ analyser, width = 480, height = 160 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let rafId;

    const drawIdle = () => {
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, width, height);
      drawGrid(ctx, width, height);
      ctx.strokeStyle = GREEN;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    if (!analyser) {
      drawIdle();
      return undefined;
    }

    const data = new Uint8Array(analyser.fftSize);

    const draw = () => {
      analyser.getByteTimeDomainData(data); // 0-255, 128 = silence

      // Fade the previous frame instead of clearing, for a phosphor-trail look
      ctx.fillStyle = 'rgba(4, 18, 10, 0.35)';
      ctx.fillRect(0, 0, width, height);
      drawGrid(ctx, width, height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = GREEN;
      ctx.shadowColor = GREEN;
      ctx.shadowBlur = 8;
      ctx.beginPath();

      const step = width / (data.length - 1);
      for (let i = 0; i < data.length; i++) {
        const x = i * step;
        const y = (data[i] / 255) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      rafId = requestAnimationFrame(draw);
    };

    draw();

    return () => cancelAnimationFrame(rafId);
  }, [analyser, width, height]);

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
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          display: 'block',
          borderRadius: '6px',
          border: '2px solid #000',
          boxShadow: `inset 0 0 20px rgba(0,0,0,0.8), 0 0 12px rgba(51,255,102,0.25)`,
        }}
      />
    </div>
  );
}
