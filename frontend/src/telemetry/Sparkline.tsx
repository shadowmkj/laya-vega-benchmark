import React from 'react';

interface SparklineProps {
  data: number[];
  maxVal?: number;
  threshold?: number; // e.g. 45ms deadline
  width?: number;
  height?: number;
}

export const Sparkline: React.FC<SparklineProps> = ({
  data,
  maxVal = 70,
  threshold = 45,
  width = 300,
  height = 50,
}) => {
  if (data.length < 2) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center border border-zinc-800 bg-zinc-950/60 text-xs text-zinc-600 font-mono"
      >
        Waiting for model decisions...
      </div>
    );
  }

  const effectiveMax = Math.max(maxVal, ...data, threshold + 10);
  const paddingY = 6;
  const innerHeight = height - paddingY * 2;

  const points = data
    .map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - paddingY - (val / effectiveMax) * innerHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const thresholdY = height - paddingY - (threshold / effectiveMax) * innerHeight;

  return (
    <div className="relative border border-zinc-800 bg-zinc-950/80 p-1">
      <svg width={width} height={height} className="overflow-visible block">
        {/* 45ms Target Deadline Line */}
        <line
          x1={0}
          y1={thresholdY}
          x2={width}
          y2={thresholdY}
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.7}
        />
        <text x={4} y={thresholdY - 3} fill="#ef4444" fontSize={9} fontFamily="monospace">
          45ms budget
        </text>

        {/* Latency Polyline */}
        <polyline
          fill="none"
          stroke="#22d3ee"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />

        {/* Recent point pulse dot */}
        {data.length > 0 && (
          <circle
            cx={width}
            cy={height - paddingY - (data[data.length - 1] / effectiveMax) * innerHeight}
            r={3.5}
            fill="#38bdf8"
          />
        )}
      </svg>
    </div>
  );
};
