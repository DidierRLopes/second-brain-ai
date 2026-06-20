// Concept: Self-Attention | Wiki: [[transformer-architecture]]

import React, { useState, useMemo } from 'react';

/**
 * Interactive Self-Attention Mechanism Explainer
 *
 * Demonstrates how transformers compute attention weights between tokens.
 * Users can explore query-key interactions, temperature effects, and the softmax computation.
 */

// Sample sentence tokenized
const SAMPLE_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];

// Learnable attention pattern (pre-defined for demo)
const LEARNED_ATTENTION_PATTERN = [
  [0.9, 0.1, 0.05, 0.02, 0.05, 0.08],
  [0.2, 0.7, 0.3, 0.15, 0.1, 0.05],
  [0.1, 0.2, 0.8, 0.4, 0.15, 0.1],
  [0.05, 0.15, 0.3, 0.7, 0.3, 0.2],
  [0.05, 0.1, 0.15, 0.2, 0.85, 0.4],
  [0.08, 0.05, 0.1, 0.15, 0.4, 0.8],
];

// Softmax function
const softmax = (scores) => {
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  return expScores.map(e => e / sumExp);
};

// Compute attention weights given raw scores and temperature
const computeAttentionRow = (rawScores, temperature) => {
  const scaledScores = rawScores.map(s => s / temperature);
  return softmax(scaledScores);
};

// Generate raw attention scores (simulated dot product)
const generateRawScores = (queryIdx, keyTokenCount, uniformMode) => {
  if (uniformMode) {
    return Array(keyTokenCount).fill(1);
  }
  // Use the learned pattern as raw scores
  return LEARNED_ATTENTION_PATTERN[queryIdx];
};

// Component for individual token pill
const TokenPill = ({ token, index, isSelected, onSelect, onHover, isHighlighted }) => (
  <button
    onClick={() => onSelect(index)}
    onMouseEnter={() => onHover(index)}
    onMouseLeave={() => onHover(null)}
    className={`
      px-3 py-2 mx-1 my-1 rounded-full font-semibold text-sm transition-all duration-200
      ${isSelected
        ? 'bg-white text-black ring-2 ring-gray-400'
        : isHighlighted
        ? 'bg-gray-500 text-white'
        : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-600'}
    `}
  >
    {token}
  </button>
);

// Heatmap cell component
const HeatmapCell = ({ value, isAttendedTo, isAttending, columnIdx, rowIdx, onCellHover }) => {
  const intensity = Math.min(value * 1.5, 1);
  const bgColor = `rgba(${Math.round(255 * (1 - intensity))}, ${Math.round(255 * (1 - intensity))}, ${Math.round(255 * (1 - intensity))}, 0.8)`;

  return (
    <div
      onMouseEnter={() => onCellHover({ row: rowIdx, col: columnIdx, value })}
      onMouseLeave={() => onCellHover(null)}
      className={`
        w-12 h-12 flex items-center justify-center text-xs font-bold
        border border-gray-600 transition-all duration-150 cursor-pointer
        ${isAttendedTo ? 'ring-2 ring-white' : ''}
        ${isAttending ? 'ring-2 ring-gray-400' : ''}
      `}
      style={{
        backgroundColor: bgColor,
        color: intensity > 0.5 ? 'black' : 'gray',
      }}
    >
      {(value * 100).toFixed(0)}
    </div>
  );
};

// Main heatmap grid component
const AttentionHeatmap = ({
  queryTokens,
  keyTokens,
  attentionMatrix,
  selectedQueryIdx,
  hoveredCellInfo,
  onCellHover,
}) => {
  return (
    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-4">Attention Matrix (Query × Key)</h3>

      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Key token headers */}
          <div className="flex mb-2">
            <div className="w-16" />
            {keyTokens.map((token, idx) => (
              <div key={idx} className="w-12 text-center text-xs font-bold text-gray-400">
                {token.slice(0, 3)}
              </div>
            ))}
          </div>

          {/* Heatmap rows */}
          {queryTokens.map((token, rowIdx) => (
            <div key={rowIdx} className="flex items-center mb-2">
              {/* Query token label */}
              <div className="w-16 text-right text-xs font-bold text-gray-400 pr-2">
                {token.slice(0, 3)}
              </div>

              {/* Cells */}
              <div className="flex gap-1">
                {attentionMatrix[rowIdx].map((value, colIdx) => (
                  <HeatmapCell
                    key={`${rowIdx}-${colIdx}`}
                    value={value}
                    isAttendedTo={selectedQueryIdx !== null && colIdx === selectedQueryIdx}
                    isAttending={selectedQueryIdx === rowIdx}
                    columnIdx={colIdx}
                    rowIdx={rowIdx}
                    onCellHover={onCellHover}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Heatmap legend */}
      <div className="mt-4 flex items-center gap-4 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-white" />
          <span>Low attention</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-gray-800 border border-gray-600" />
          <span>High attention</span>
        </div>
        {hoveredCellInfo && (
          <div className="ml-4 text-gray-300">
            Query: <span className="font-bold">{queryTokens[hoveredCellInfo.row]}</span> →
            Key: <span className="font-bold">{keyTokens[hoveredCellInfo.col]}</span>
            Weight: <span className="font-bold">{(hoveredCellInfo.value * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Softmax computation panel
const SoftmaxPanel = ({ selectedQueryIdx, rawScores, temperature, attentionWeights }) => {
  if (selectedQueryIdx === null) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h3 className="text-lg font-bold text-white mb-3">Softmax Computation</h3>
        <p className="text-gray-400 text-sm">
          Click on a token above to see how raw attention scores are converted to attention weights.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-4">Softmax Computation (Query Token: {SAMPLE_TOKENS[selectedQueryIdx]})</h3>

      <div className="space-y-4 text-sm">
        {/* Step 1: Raw scores */}
        <div>
          <p className="text-gray-300 font-semibold mb-2">Step 1: Raw Scores (dot product)</p>
          <div className="bg-gray-900 p-3 rounded border border-gray-700">
            <div className="flex flex-wrap gap-2">
              {rawScores.map((score, idx) => (
                <span key={idx} className="text-xs text-gray-300">
                  <span className="text-gray-400">{SAMPLE_TOKENS[idx]}</span>: {score.toFixed(3)}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Step 2: Temperature scaling */}
        <div>
          <p className="text-gray-300 font-semibold mb-2">Step 2: Temperature Scaling (÷ {temperature})</p>
          <div className="bg-gray-900 p-3 rounded border border-gray-700">
            <div className="flex flex-wrap gap-2">
              {rawScores.map((score, idx) => {
                const scaled = score / temperature;
                return (
                  <span key={idx} className="text-xs text-gray-300">
                    <span className="text-gray-400">{SAMPLE_TOKENS[idx]}</span>: {scaled.toFixed(3)}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        {/* Step 3: Exp */}
        <div>
          <p className="text-gray-300 font-semibold mb-2">Step 3: Exponential (e^x)</p>
          <div className="bg-gray-900 p-3 rounded border border-gray-700">
            <div className="flex flex-wrap gap-2">
              {rawScores.map((score, idx) => {
                const scaled = score / temperature;
                const exp = Math.exp(scaled);
                return (
                  <span key={idx} className="text-xs text-gray-300">
                    <span className="text-gray-400">{SAMPLE_TOKENS[idx]}</span>: {exp.toFixed(4)}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        {/* Step 4: Normalize */}
        <div>
          <p className="text-gray-300 font-semibold mb-2">Step 4: Normalize (÷ sum)</p>
          <div className="bg-gray-900 p-3 rounded border border-gray-700">
            <div className="flex flex-wrap gap-2">
              {attentionWeights.map((weight, idx) => (
                <span key={idx} className="text-xs text-gray-300">
                  <span className="text-gray-400">{SAMPLE_TOKENS[idx]}</span>: {(weight * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Formula */}
        <div className="bg-gray-900 p-3 rounded border border-gray-600 mt-4">
          <p className="text-gray-300 text-xs leading-relaxed font-mono">
            attention(Q, K, V) = softmax(Q·K<sup>T</sup> / √d<sub>k</sub>) · V
          </p>
          <p className="text-gray-400 text-xs mt-2">
            Temperature acts like √d<sub>k</sub>: lower = sharper, higher = softer
          </p>
        </div>
      </div>
    </div>
  );
};

// Explanation panel
const ExplanationPanel = ({ selectedQueryIdx, uniformMode, temperature, hoveredCellInfo }) => {
  let explanation = '';

  if (selectedQueryIdx !== null && hoveredCellInfo) {
    const weight = (hoveredCellInfo.value * 100).toFixed(1);
    explanation = `"${SAMPLE_TOKENS[selectedQueryIdx]}" attends to "${SAMPLE_TOKENS[hoveredCellInfo.col]}" with ${weight}% weight. This is computed via softmax over all key tokens.`;
  } else if (selectedQueryIdx !== null) {
    explanation = `"${SAMPLE_TOKENS[selectedQueryIdx]}" is the query token. Its attention weights across all key tokens are shown in the heatmap. The heatmap row highlights where this token focuses.`;
  } else if (uniformMode) {
    explanation = 'Uniform attention: All tokens attend equally to all other tokens (1/6 each). This is the baseline—real attention learns meaningful patterns.';
  } else {
    explanation = 'Learned attention: The model has learned which tokens are important for each position. Click a token to see its attention pattern.';
  }

  return (
    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-2">What's Happening?</h3>
      <p className="text-gray-100 leading-relaxed">{explanation}</p>
      {selectedQueryIdx !== null && (
        <p className="text-gray-300 text-sm mt-3 italic">
          Temperature: {temperature.toFixed(2)} — {
            temperature < 1 ? 'Sharp distribution (model is confident)' :
            temperature > 1 ? 'Soft distribution (model is uncertain)' :
            'Default distribution'
          }
        </p>
      )}
    </div>
  );
};

// Main component
export default function AttentionMechanismExplainer() {
  const [selectedQueryIdx, setSelectedQueryIdx] = useState(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState(null);
  const [temperature, setTemperature] = useState(1.0);
  const [uniformMode, setUniformMode] = useState(false);
  const [hoveredCellInfo, setHoveredCellInfo] = useState(null);

  // Compute attention matrix
  const attentionMatrix = useMemo(() => {
    return SAMPLE_TOKENS.map((_, queryIdx) => {
      const rawScores = generateRawScores(queryIdx, SAMPLE_TOKENS.length, uniformMode);
      return computeAttentionRow(rawScores, temperature);
    });
  }, [temperature, uniformMode]);

  // Get attention weights for selected token
  const selectedAttentionWeights = selectedQueryIdx !== null
    ? attentionMatrix[selectedQueryIdx]
    : null;

  const selectedRawScores = selectedQueryIdx !== null
    ? generateRawScores(selectedQueryIdx, SAMPLE_TOKENS.length, uniformMode)
    : null;

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold mb-2">Self-Attention Mechanism</h1>
          <p className="text-gray-400 text-lg">
            Explore how transformer models determine which tokens are important to each other.
          </p>
        </div>

        {/* Controls */}
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 space-y-6">
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Input Tokens</h2>
            <div className="flex flex-wrap">
              {SAMPLE_TOKENS.map((token, idx) => (
                <TokenPill
                  key={idx}
                  token={token}
                  index={idx}
                  isSelected={selectedQueryIdx === idx}
                  onSelect={setSelectedQueryIdx}
                  onHover={setHoveredTokenIdx}
                  isHighlighted={hoveredTokenIdx === idx}
                />
              ))}
            </div>
            <p className="text-gray-400 text-sm mt-3">
              Click a token to see which other tokens it attends to.
            </p>
          </div>

          {/* Toggle for uniform vs learned */}
          <div className="border-t border-gray-700 pt-6">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={uniformMode}
                onChange={(e) => {
                  setUniformMode(e.target.checked);
                  setSelectedQueryIdx(null);
                }}
                className="w-5 h-5 accent-white"
              />
              <span className="text-white font-semibold">
                {uniformMode ? 'Uniform Attention' : 'Learned Attention'}
              </span>
            </label>
            <p className="text-gray-400 text-sm mt-2">
              Toggle to see the difference between random attention and learned patterns.
            </p>
          </div>

          {/* Temperature slider */}
          <div className="border-t border-gray-700 pt-6">
            <div className="flex items-center justify-between mb-3">
              <label className="text-white font-semibold">Temperature: {temperature.toFixed(2)}</label>
              <span className="text-xs text-gray-400">
                {temperature < 0.5 ? 'Sharp' : temperature > 1.5 ? 'Diffuse' : 'Balanced'}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-white cursor-pointer"
            />
            <p className="text-gray-400 text-sm mt-2">
              Lower temperature = sharper attention (high confidence).
              Higher temperature = softer attention (more uniform).
            </p>
          </div>
        </div>

        {/* Explanation panel */}
        <ExplanationPanel
          selectedQueryIdx={selectedQueryIdx}
          uniformMode={uniformMode}
          temperature={temperature}
          hoveredCellInfo={hoveredCellInfo}
        />

        {/* Attention heatmap */}
        <AttentionHeatmap
          queryTokens={SAMPLE_TOKENS}
          keyTokens={SAMPLE_TOKENS}
          attentionMatrix={attentionMatrix}
          selectedQueryIdx={selectedQueryIdx}
          hoveredCellInfo={hoveredCellInfo}
          onCellHover={setHoveredCellInfo}
        />

        {/* Softmax computation panel */}
        {selectedQueryIdx !== null && selectedAttentionWeights && (
          <SoftmaxPanel
            selectedQueryIdx={selectedQueryIdx}
            rawScores={selectedRawScores}
            temperature={temperature}
            attentionWeights={selectedAttentionWeights}
          />
        )}

        {/* Footer explanation */}
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 space-y-4">
          <h3 className="text-lg font-bold text-white">Key Concepts</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="text-white font-semibold mb-2">Query (Q)</h4>
              <p className="text-gray-400">
                The current token asking "What should I attend to?"
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Key (K)</h4>
              <p className="text-gray-400">
                All tokens that could be attended to. Matched with query via dot product.
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Softmax</h4>
              <p className="text-gray-400">
                Converts raw scores to a probability distribution (0-1, sum to 1).
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Temperature</h4>
              <p className="text-gray-400">
                Controls sharpness: lower = pick one, higher = spread attention evenly.
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Attention Matrix</h4>
              <p className="text-gray-400">
                Rows = queries, columns = keys. Each cell shows how much one token attends to another.
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-2">Self-Attention</h4>
              <p className="text-gray-400">
                Query, key, and value come from the same sequence. Tokens attend to themselves and neighbors.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
