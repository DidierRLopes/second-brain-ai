// Concept: Quantization | Wiki: [[quantization-fundamentals]] [[quantization-methods]] [[activation-outliers]]

import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

// Generate normally distributed weights with one outlier
const generateWeights = () => {
  const weights = [];
  // Generate ~19 normal weights from normal distribution
  for (let i = 0; i < 19; i++) {
    let val = 0;
    for (let j = 0; j < 6; j++) {
      val += Math.random();
    }
    weights.push((val - 3) * 0.3); // Mean 0, std ~0.3
  }
  // Add one large outlier
  weights.push(2.8);
  return weights.sort((a, b) => a - b);
};

// Quantize a value to a given bit width
const quantize = (value, bitWidth, min, max, isOutlier = false) => {
  if (bitWidth === 32) return value;

  const range = max - min;
  const levels = Math.pow(2, bitWidth);
  const step = range / (levels - 1);

  // Normalize to [0, 1]
  const normalized = (value - min) / range;
  // Quantize
  const quantizedNorm = Math.round(normalized * (levels - 1)) / (levels - 1);
  // Denormalize
  return min + quantizedNorm * range;
};

// Separate outlier from normal weights
const separateOutlier = (weights) => {
  const sorted = [...weights].sort((a, b) => a - b);
  const max = sorted[sorted.length - 1];
  const threshold = 1.5; // threshold for outlier
  const outliers = weights.filter((w) => w > threshold);
  const normals = weights.filter((w) => w <= threshold);
  return { outliers, normals };
};

export default function QuantizationExplainer() {
  const [bitWidth, setBitWidth] = useState(8);
  const [enableOutlierMode, setEnableOutlierMode] = useState(false);

  const weights = useMemo(() => generateWeights(), []);

  // Compute ranges
  const minWeight = Math.min(...weights);
  const maxWeight = Math.max(...weights);
  const range = maxWeight - minWeight;

  // If outlier mode, separate and requantize
  let quantizedWeights, outlierValue, normalMin, normalMax;

  if (enableOutlierMode) {
    const { outliers, normals } = separateOutlier(weights);
    outlierValue = outliers.length > 0 ? outliers[0] : null;

    if (outlierValue && normals.length > 0) {
      normalMin = Math.min(...normals);
      normalMax = Math.max(...normals);

      quantizedWeights = weights.map((w) => {
        if (w === outlierValue) {
          return w; // Keep outlier as-is (or could store separately)
        }
        return quantize(w, bitWidth, normalMin, normalMax);
      });
    } else {
      quantizedWeights = weights.map((w) =>
        quantize(w, bitWidth, minWeight, maxWeight)
      );
    }
  } else {
    quantizedWeights = weights.map((w) =>
      quantize(w, bitWidth, minWeight, maxWeight)
    );
  }

  // Compute errors
  const errors = weights.map((orig, i) => Math.abs(orig - quantizedWeights[i]));
  const maxError = Math.max(...errors);
  const meanError = errors.reduce((a, b) => a + b, 0) / errors.length;

  // Memory usage calculation
  const originalBits = weights.length * 32;
  const quantizedBits = weights.length * bitWidth;
  const memoryReduction = ((1 - quantizedBits / originalBits) * 100).toFixed(1);

  // Representable values
  const representableValues = Math.pow(2, bitWidth);

  // Prepare data for charts
  const numberLineData = weights.map((w, i) => ({
    index: i,
    original: w,
    quantized: quantizedWeights[i],
    error: errors[i],
    isOutlier: enableOutlierMode && w > 1.5,
  }));

  const errorData = weights.map((w, i) => ({
    index: i,
    error: errors[i],
  }));

  // Description based on bit width
  const getDescription = () => {
    const descriptions = {
      32: 'Full precision (32-bit float): Every weight uses a full 32-bit floating point. No quantization error.',
      16: '16-bit half precision: Roughly half the memory. Suitable for most inference tasks. Small quantization error.',
      8: '8-bit integer quantization: 75% memory reduction. Common for on-device inference. Moderate quantization error.',
      4: '4-bit quantization: 87.5% memory reduction. Aggressive compression for edge devices. Significant quantization error.',
      2: '2-bit quantization: Extreme compression (93.75% reduction). Severe quantization error but enables rapid inference.',
    };
    return descriptions[bitWidth] || '';
  };

  const getOutlierExplanation = () => {
    if (!enableOutlierMode) {
      return 'Notice the largest weight (right side). It dominates the quantization range, forcing all other weights to share fewer representable values, increasing their quantization error.';
    }
    return 'Outlier-aware mode: The largest weight is separated and stored at full precision. Normal weights are quantized within their own range, dramatically reducing quantization error for the majority of weights.';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Quantization & Bit Reduction</h1>
          <p className="text-gray-400">
            Interactive exploration of how LLMs reduce model size through weight quantization.
          </p>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* Bit Width Slider */}
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="mb-6">
              <label className="block text-lg font-semibold mb-4">
                Bit Width: <span className="text-white">{bitWidth}-bit</span>
              </label>
              <input
                type="range"
                min="2"
                max="32"
                step="1"
                value={bitWidth}
                onChange={(e) => setBitWidth(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-400 mt-2">
                <span>2-bit</span>
                <span>32-bit</span>
              </div>
            </div>

            {/* Quick select buttons */}
            <div className="flex gap-2">
              {[2, 4, 8, 16, 32].map((bit) => (
                <button
                  key={bit}
                  onClick={() => setBitWidth(bit)}
                  className={`px-3 py-2 rounded font-semibold transition-colors ${
                    bitWidth === bit
                      ? 'bg-white text-black'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                  }`}
                >
                  {bit}-bit
                </button>
              ))}
            </div>
          </div>

          {/* Outlier Mode Toggle */}
          <div className="bg-gray-800 rounded-lg p-6">
            <label className="flex items-center gap-4 cursor-pointer">
              <input
                type="checkbox"
                checked={enableOutlierMode}
                onChange={(e) => setEnableOutlierMode(e.target.checked)}
                className="w-6 h-6"
              />
              <span className="text-lg font-semibold">
                Enable Outlier-Aware Quantization
                <br />
                <span className="text-sm text-gray-400">
                  (LLM.int8() technique)
                </span>
              </span>
            </label>
            <p className="mt-4 text-sm text-gray-300">
              Separates extreme outlier weights and quantizes normal weights
              independently, reducing quantization error dramatically.
            </p>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-300">Memory Usage</div>
            <div className="text-2xl font-bold text-white">
              {quantizedBits} bits
            </div>
            <div className="text-xs text-gray-400">
              {memoryReduction}% reduction
            </div>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-300">Representable Values</div>
            <div className="text-2xl font-bold text-white">
              {representableValues}
            </div>
            <div className="text-xs text-gray-400">from {weights.length} weights</div>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-300">Max Error</div>
            <div className="text-2xl font-bold text-white">
              {maxError.toFixed(4)}
            </div>
            <div className="text-xs text-gray-400">
              {((maxError / (maxWeight - minWeight)) * 100).toFixed(2)}% of range
            </div>
          </div>

          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-300">Mean Error</div>
            <div className="text-2xl font-bold text-white">
              {meanError.toFixed(4)}
            </div>
            <div className="text-xs text-gray-400">average per weight</div>
          </div>
        </div>

        {/* Description */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-3 text-white">What's Happening</h2>
          <p className="text-gray-300 mb-4">{getDescription()}</p>
          <p className="text-gray-300">{getOutlierExplanation()}</p>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Weight Distribution Chart */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Weight Quantization</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={numberLineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                <XAxis dataKey="index" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #555',
                  }}
                  formatter={(value) => value.toFixed(4)}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="original"
                  stroke="#888"
                  dot={{ r: 3 }}
                  name="Original (32-bit)"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="quantized"
                  stroke="#ffffff"
                  dot={{ r: 3 }}
                  strokeWidth={2}
                  name={`Quantized (${bitWidth}-bit)`}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Quantization Error Chart */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Quantization Error</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={errorData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                <XAxis dataKey="index" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #555',
                  }}
                  formatter={(value) => value.toFixed(4)}
                />
                <Bar dataKey="error" fill="#d1d5db" name="Error" radius={[4, 4, 0, 0]}>
                  {errorData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.error > maxError * 0.5 ? '#f3f4f6' : '#d1d5db'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Detailed Explanation */}
        <div className="bg-gray-800 rounded-lg p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Understanding Quantization</h2>

          <div className="space-y-3 text-gray-300 text-sm">
            <div>
              <strong className="text-white font-medium">Quantization:</strong> Converting
              high-precision weights (32-bit floats) into lower-precision
              representations (8-bit, 4-bit, etc.). This reduces model size and
              memory requirements.
            </div>

            <div>
              <strong className="text-white font-medium">Trade-off:</strong> Memory
              savings vs. accuracy loss. Lower bit widths compress more but
              introduce larger quantization errors.
            </div>

            <div>
              <strong className="text-white font-medium">Outliers:</strong> Some LLM
              weights are much larger than the rest. These dominance the
              quantization range, forcing normal weights into fewer representable
              values. Outlier-aware techniques (LLM.int8) handle them separately.
            </div>

            <div>
              <strong className="text-white font-medium">Representable Values:</strong>{' '}
              An n-bit integer can represent 2^n distinct values. Weights are
              snapped to the nearest representable value.
            </div>

            <div>
              <strong className="text-white font-medium">Use Cases:</strong> 4-8 bit
              quantization is common for production LLM inference (faster,
              cheaper, mobile). 2-bit is extreme but emerging for ultra-light
              models.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
