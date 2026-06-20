// Concept: LoRA | Wiki: [[parameter-efficient-fine-tuning]]

import React, { useState, useMemo } from 'react';

/**
 * Heatmap color mapping: black (negative) -> gray (zero) -> white (positive)
 */
function getHeatmapColor(value, min, max) {
  const range = max - min;
  const normalized = (value - min) / (range || 1); // 0 to 1

  if (normalized < 0.5) {
    // Black to gray
    const t = normalized * 2; // 0 to 1
    const gray = Math.round(128 * t); // 0 to 128
    return `rgb(${gray}, ${gray}, ${gray})`;
  } else {
    // Gray to white
    const t = (normalized - 0.5) * 2; // 0 to 1
    const gray = Math.round(128 + 127 * t); // 128 to 255
    return `rgb(${gray}, ${gray}, ${gray})`;
  }
}

/**
 * Initialize random matrix
 */
function randomMatrix(rows, cols, seed = 0) {
  // Simple seeded random number generator for reproducibility
  const seededRandom = (index) => {
    const x = Math.sin(index + seed) * 10000;
    return x - Math.floor(x);
  };

  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => (seededRandom(i * cols + j) - 0.5) * 2)
  );
}

/**
 * Matrix multiplication: (m x n) * (n x p) = (m x p)
 */
function matmul(A, B) {
  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;

  const result = Array.from({ length: m }, () => Array(p).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

/**
 * Matrix subtraction: A - B
 */
function matsubtract(A, B) {
  return A.map((row, i) =>
    row.map((val, j) => val - B[i][j])
  );
}

/**
 * Frobenius norm: sqrt(sum of squared elements)
 */
function frobeniusNorm(matrix) {
  let sum = 0;
  for (let row of matrix) {
    for (let val of row) {
      sum += val * val;
    }
  }
  return Math.sqrt(sum);
}

/**
 * Get min and max values in matrix
 */
function getMatrixMinMax(matrix) {
  let min = Infinity;
  let max = -Infinity;
  for (let row of matrix) {
    for (let val of row) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }
  return { min, max };
}

/**
 * MatrixDisplay component
 */
function MatrixDisplay({ matrix, label, cellSize = 30, title = '' }) {
  const { min, max } = getMatrixMinMax(matrix);
  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;

  return (
    <div className="flex flex-col items-center gap-2">
      {title && <h4 className="text-sm font-semibold text-gray-300">{title}</h4>}
      <div className="inline-block border border-gray-600 rounded">
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: '1px', padding: '4px', backgroundColor: '#1a1a1a' }}>
          {matrix.map((row, i) =>
            row.map((val, j) => (
              <div
                key={`${i}-${j}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getHeatmapColor(val, min, max),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '9px',
                  fontWeight: '500',
                  color: Math.abs(val) > (max + min) / 2 ? '#000' : '#fff',
                }}
                title={val.toFixed(2)}
              >
                {cellSize > 25 ? val.toFixed(1) : ''}
              </div>
            ))
          )}
        </div>
      </div>
      <p className="text-xs text-gray-400">
        {label} {rows}×{cols}
      </p>
    </div>
  );
}

/**
 * Main LoRA component
 */
export default function LoRAExplainer() {
  const [rank, setRank] = useState(4);
  const [seed, setSeed] = useState(0);
  const d = 8; // embedding dimension

  // Generate matrices
  const { W, B, A, dW, W_prime, reconstructionError, paramStats } = useMemo(() => {
    const W = randomMatrix(d, d, seed);
    const B = randomMatrix(d, rank, seed + 1);
    const A = randomMatrix(rank, d, seed + 2);

    // Compute ΔW = B × A
    const dW = matmul(B, A);

    // Compute W' = W + ΔW
    const W_prime = W.map((row, i) =>
      row.map((val, j) => val + dW[i][j])
    );

    // Reconstruction error: ||W_actual - W_approx||_F
    const reconstructionError = frobeniusNorm(matsubtract(dW, dW));

    // Parameter count
    const wParams = d * d;
    const loraParams = d * rank + rank * d;
    const compressionRatio = (wParams / loraParams).toFixed(2);
    const trainablePercent = ((loraParams / wParams) * 100).toFixed(1);
    const memorySavings = (100 - (loraParams / wParams) * 100).toFixed(1);

    return {
      W,
      B,
      A,
      dW,
      W_prime,
      reconstructionError: 0, // B×A approximates a delta, error is structural
      paramStats: {
        wParams,
        loraParams,
        compressionRatio,
        trainablePercent,
        memorySavings,
      },
    };
  }, [rank, seed]);

  const handleRandomize = () => {
    setSeed(s => s + 1);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">LoRA: Low-Rank Adaptation</h1>
          <p className="text-gray-400 text-lg">
            Interactive visualization of how low-rank matrix decomposition enables efficient fine-tuning
          </p>
        </div>

        {/* Rank Slider */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <div className="flex items-center gap-4">
            <label className="text-lg font-semibold min-w-32">Rank (r):</label>
            <input
              type="range"
              min="1"
              max="8"
              value={rank}
              onChange={(e) => setRank(Number(e.target.value))}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #6b7280 0%, #6b7280 ${(rank / 8) * 100}%, #374151 ${(rank / 8) * 100}%, #374151 100%)`
              }}
            />
            <span className="text-2xl font-bold text-white min-w-12">{rank}</span>
          </div>
          <p className="text-sm text-gray-400 mt-3">
            Adjust the rank to control the approximation capacity. Higher rank = larger parameter count but better approximation.
          </p>
        </div>

        {/* Explanation Text */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-8">
          <h3 className="font-semibold mb-2 text-white">How LoRA Works:</h3>
          <p className="text-gray-200 text-sm leading-relaxed">
            Instead of fine-tuning all {paramStats.wParams} parameters in the weight matrix W, LoRA decomposes the weight update
            into two smaller matrices: <span className="font-mono text-white">B ({d}×{rank})</span> and{' '}
            <span className="font-mono text-white">A ({rank}×{d})</span>. The low-rank update is computed as{' '}
            <span className="font-mono text-white">ΔW = B × A</span>, requiring only{' '}
            <span className="font-mono font-bold text-white">{paramStats.loraParams} trainable parameters</span> ({paramStats.trainablePercent}% of original).
            This achieves a {paramStats.compressionRatio}× compression ratio while preserving most of the model's expressiveness.
          </p>
        </div>

        {/* Main Visualization */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mb-8">
          {/* Frozen W */}
          <div className="flex flex-col items-center">
            <div className="bg-gray-800 p-4 rounded-lg border border-gray-600 w-full flex justify-center opacity-60">
              <MatrixDisplay matrix={W} label="Frozen" cellSize={24} title="W (Frozen)" />
            </div>
            <p className="text-xs text-gray-500 mt-2">Not fine-tuned</p>
          </div>

          {/* B Matrix */}
          <div className="flex flex-col items-center">
            <div className="bg-gray-800 p-4 rounded-lg border-2 border-gray-600 w-full flex justify-center">
              <MatrixDisplay matrix={B} label="Trainable" cellSize={24} title="B (Trainable)" />
            </div>
            <p className="text-xs text-gray-400 mt-2">{d}×{rank} = {d * rank} params</p>
          </div>

          {/* Multiplication Symbol and A Matrix */}
          <div className="flex flex-col items-center justify-center lg:mt-12">
            <div className="text-5xl font-bold text-gray-500 mb-4">×</div>
            <div className="bg-gray-800 p-4 rounded-lg border-2 border-gray-600 w-full flex justify-center">
              <MatrixDisplay matrix={A} label="Trainable" cellSize={24} title="A (Trainable)" />
            </div>
            <p className="text-xs text-gray-400 mt-2">{rank}×{d} = {rank * d} params</p>
          </div>

          {/* Result ΔW */}
          <div className="flex flex-col items-center">
            <div className="bg-gray-800 p-4 rounded-lg border-2 border-gray-600 w-full flex justify-center">
              <MatrixDisplay matrix={dW} label="Update" cellSize={24} title="ΔW = B×A" />
            </div>
            <p className="text-xs text-gray-400 mt-2">Weight update matrix</p>
          </div>
        </div>

        {/* Final Result: W' = W + ΔW */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4">Final Adapted Weights: W' = W + ΔW</h3>
          <div className="flex justify-center">
            <MatrixDisplay matrix={W_prime} label="Adapted weights" cellSize={28} title="W' (Final)" />
          </div>
        </div>

        {/* Stats Panel */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-300 text-sm mb-1">Original W Params</p>
            <p className="text-3xl font-bold text-white">{paramStats.wParams}</p>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-300 text-sm mb-1">Trainable (B+A)</p>
            <p className="text-3xl font-bold text-white">{paramStats.loraParams}</p>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-300 text-sm mb-1">% Trainable</p>
            <p className="text-3xl font-bold text-white">{paramStats.trainablePercent}%</p>
          </div>

          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <p className="text-gray-300 text-sm mb-1">Memory Savings</p>
            <p className="text-3xl font-bold text-white">{paramStats.memorySavings}%</p>
          </div>
        </div>

        {/* Details Table */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700 overflow-x-auto">
          <h3 className="text-lg font-semibold mb-4">Parameter Breakdown</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-600">
                <th className="text-left py-2 px-3 text-gray-300">Component</th>
                <th className="text-left py-2 px-3 text-gray-300">Shape</th>
                <th className="text-right py-2 px-3 text-gray-300">Parameters</th>
                <th className="text-left py-2 px-3 text-gray-300">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-700 bg-gray-700 bg-opacity-30">
                <td className="py-3 px-3 font-mono text-white">W</td>
                <td className="py-3 px-3 font-mono text-gray-300">{d}×{d}</td>
                <td className="py-3 px-3 text-right font-mono text-white">{paramStats.wParams}</td>
                <td className="py-3 px-3 text-gray-400">Frozen</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-3 px-3 font-mono text-white">B</td>
                <td className="py-3 px-3 font-mono text-gray-300">{d}×{rank}</td>
                <td className="py-3 px-3 text-right font-mono text-white">{d * rank}</td>
                <td className="py-3 px-3 text-gray-400">Trainable</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-3 px-3 font-mono text-white">A</td>
                <td className="py-3 px-3 font-mono text-gray-300">{rank}×{d}</td>
                <td className="py-3 px-3 text-right font-mono text-white">{rank * d}</td>
                <td className="py-3 px-3 text-gray-400">Trainable</td>
              </tr>
              <tr className="bg-gray-700 bg-opacity-40 font-semibold">
                <td className="py-3 px-3">Total Trainable</td>
                <td className="py-3 px-3">—</td>
                <td className="py-3 px-3 text-right text-white">{paramStats.loraParams}</td>
                <td className="py-3 px-3 text-white">{paramStats.compressionRatio}× smaller</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 mb-8">
          <button
            onClick={handleRandomize}
            className="bg-white text-black hover:bg-gray-200 font-semibold py-3 px-6 rounded-lg transition-all duration-200 shadow-lg"
          >
            Randomize Matrices
          </button>
        </div>

        {/* Footer Info */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-xs text-gray-400">
          <p>
            <strong>Concept:</strong> LoRA reduces fine-tuning cost by decomposing weight updates into low-rank matrices.
            Instead of updating all parameters, only B and A are trained, then combined via matrix multiplication.
          </p>
        </div>
      </div>
    </div>
  );
}
