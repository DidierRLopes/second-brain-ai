// Concept: KV Cache | Wiki: [[kv-cache]] [[attention-variants]] [[inference-optimization]]

import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function KVCacheExplainer() {
  // Configuration
  const NUM_LAYERS = 32; // typical transformer depth
  const HIDDEN_DIM = 4096;
  const NUM_HEADS = 32;
  const VRAM_LIMIT_MB = 24000; // 24GB typical GPU limit
  const BYTES_PER_PARAM = 2; // float16

  // Attention type configurations
  const ATTENTION_CONFIGS = {
    MHA: {
      name: 'Multi-Head Attention (MHA)',
      kvHeads: NUM_HEADS,
      description: 'Full KV cache per head. Baseline approach.',
    },
    MQA: {
      name: 'Multi-Query Attention (MQA)',
      kvHeads: 1,
      description: 'Single KV head shared across all query heads. Minimal cache.',
    },
    GQA: {
      name: 'Grouped Query Attention (GQA)',
      kvHeads: 8, // typical: 8 groups
      description: 'KV heads shared across groups of query heads. Balanced approach.',
    },
  };

  // State
  const [tokensGenerated, setTokensGenerated] = useState(0);
  const [attentionType, setAttentionType] = useState('MHA');
  const [memoryHistory, setMemoryHistory] = useState([
    { token: 0, memoryMB: 0, attentionType: 'MHA' },
  ]);

  // Calculate KV cache size
  const calculateCacheSize = (numTokens, attentionType) => {
    const config = ATTENTION_CONFIGS[attentionType];
    const headDim = HIDDEN_DIM / NUM_HEADS;

    // Per layer: num_tokens * kv_heads * head_dim * 2 (K and V)
    const perLayerTokens = numTokens * config.kvHeads * headDim;
    const perLayerBytes = perLayerTokens * BYTES_PER_PARAM * 2; // *2 for K and V
    const totalBytes = perLayerBytes * NUM_LAYERS;
    const totalMB = totalBytes / (1024 * 1024);

    return {
      totalMB,
      perLayerKB: (perLayerBytes / 1024).toFixed(2),
      totalEntries: numTokens * config.kvHeads * NUM_LAYERS,
    };
  };

  const currentConfig = ATTENTION_CONFIGS[attentionType];
  const currentStats = calculateCacheSize(tokensGenerated, attentionType);

  // Generate next token handler
  const handleGenerateToken = () => {
    const newTokenCount = tokensGenerated + 1;
    setTokensGenerated(newTokenCount);

    const newStats = calculateCacheSize(newTokenCount, attentionType);
    setMemoryHistory([
      ...memoryHistory,
      { token: newTokenCount, memoryMB: newStats.totalMB, attentionType },
    ]);
  };

  // Handle attention type change
  const handleAttentionChange = (e) => {
    const newType = e.target.value;
    setAttentionType(newType);

    // Recalculate memory history with new attention type
    const updatedHistory = memoryHistory.map((entry) => {
      const stats = calculateCacheSize(entry.token, newType);
      return {
        ...entry,
        memoryMB: stats.totalMB,
        attentionType: newType,
      };
    });
    setMemoryHistory(updatedHistory);
  };

  // Reset handler
  const handleReset = () => {
    setTokensGenerated(0);
    setMemoryHistory([{ token: 0, memoryMB: 0, attentionType }]);
  };

  // Sample tokens for visualization
  const sampleTokens = [
    'The',
    'quick',
    'brown',
    'fox',
    'jumps',
    'over',
    'the',
    'lazy',
    'dog',
  ];
  const generatedText = sampleTokens.slice(0, tokensGenerated + 1).join(' ');

  // Calculate memory percentage
  const memoryPercentage = Math.min(
    100,
    (currentStats.totalMB / VRAM_LIMIT_MB) * 100
  );

  return (
    <div className="w-full min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">KV Cache Growth in LLM Inference</h1>
          <p className="text-gray-400 text-lg">
            Visualize how the KV cache grows with each generated token and how different
            attention mechanisms affect memory consumption.
          </p>
        </div>

        {/* Main grid layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Left panel: Controls and Generated Text */}
          <div className="lg:col-span-1 space-y-6">
            {/* Generated Text */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase">
                Generated Text
              </h3>
              <div className="bg-gray-900 p-3 rounded border border-gray-700 min-h-20 flex items-center">
                <p className="text-lg leading-relaxed">
                  {generatedText || <span className="text-gray-500 italic">No tokens generated</span>}
                </p>
              </div>
            </div>

            {/* Attention Type Selector */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase">
                Attention Type
              </h3>
              <select
                value={attentionType}
                onChange={handleAttentionChange}
                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white focus:border-gray-500 focus:outline-none"
              >
                {Object.entries(ATTENTION_CONFIGS).map(([key, config]) => (
                  <option key={key} value={key}>
                    {config.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-400 mt-2">{currentConfig.description}</p>
            </div>

            {/* Quick Stats */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
              <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase">Stats</h3>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-gray-400">Tokens Generated</span>
                  <span className="font-mono font-bold">{tokensGenerated}</span>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-gray-400">Cache Entries</span>
                  <span className="font-mono font-bold">
                    {currentStats.totalEntries.toLocaleString()}
                  </span>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-gray-400">Cache Size</span>
                  <span className="font-mono font-bold text-white">
                    {currentStats.totalMB.toFixed(1)} MB
                  </span>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-gray-400">VRAM Usage</span>
                  <span className="font-mono font-bold">
                    {memoryPercentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <button
                onClick={handleGenerateToken}
                disabled={tokensGenerated >= sampleTokens.length - 1}
                className={`w-full py-3 px-4 rounded font-semibold transition ${
                  tokensGenerated >= sampleTokens.length - 1
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-white text-black hover:bg-gray-200'
                }`}
              >
                Generate Next Token
              </button>
              <button
                onClick={handleReset}
                className="w-full py-3 px-4 bg-gray-700 hover:bg-gray-600 text-white rounded font-semibold transition"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Right panel: Charts and Memory Bar */}
          <div className="lg:col-span-2 space-y-6">
            {/* Memory Usage Bar */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-4 uppercase">
                GPU VRAM Usage
              </h3>
              <div className="space-y-2">
                <div className="bg-gray-800 rounded h-8 border border-gray-600 overflow-hidden">
                  <div
                    className="h-full transition-all duration-300 bg-gray-300"
                    style={{ width: `${Math.min(memoryPercentage, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm text-gray-400">
                  <span>0 MB</span>
                  <span>
                    {currentStats.totalMB.toFixed(1)} MB / {VRAM_LIMIT_MB} MB
                  </span>
                </div>
              </div>
            </div>

            {/* Memory Chart */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-4 uppercase">
                Memory Growth Over Time
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={memoryHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="token"
                    stroke="#9ca3af"
                    label={{ value: 'Token #', position: 'insideBottomRight', offset: -5 }}
                  />
                  <YAxis
                    stroke="#9ca3af"
                    label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                    formatter={(value) => value.toFixed(1)}
                  />
                  <ReferenceLine
                    y={VRAM_LIMIT_MB}
                    stroke="#ffffff"
                    strokeDasharray="5 5"
                    label={{
                      value: 'VRAM Limit (24 GB)',
                      position: 'right',
                      fill: '#ffffff',
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="memoryMB"
                    stroke="#ffffff"
                    dot={{ fill: '#ffffff', r: 4 }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* KV Cache Grid Visualization */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">KV Cache Structure</h2>
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 overflow-x-auto">
            <p className="text-gray-400 text-sm mb-4">
              Each cell represents a K,V pair for one token at one layer. Rows = {NUM_LAYERS} layers,
              Columns = {tokensGenerated + 1} token(s). KV Heads = {currentConfig.kvHeads}
            </p>
            <div className="flex gap-1 flex-wrap">
              {/* Show layers (abbreviated for space) */}
              {Array.from({ length: Math.min(NUM_LAYERS, 8) }).map((_, layerIdx) => (
                <div key={layerIdx} className="flex gap-1">
                  {Array.from({ length: tokensGenerated + 1 }).map((_, tokenIdx) => (
                    <div
                      key={`${layerIdx}-${tokenIdx}`}
                      className={`w-6 h-6 border rounded-sm text-center text-xs flex items-center justify-center transition ${
                        tokenIdx === tokensGenerated
                          ? 'bg-white border-gray-400'
                          : 'bg-gray-600 border-gray-500'
                      }`}
                      title={`Layer ${layerIdx}, Token ${tokenIdx}`}
                    >
                      {currentConfig.kvHeads > 1 ? 'M' : '1'}
                    </div>
                  ))}
                </div>
              ))}
              {NUM_LAYERS > 8 && (
                <div className="text-gray-500 text-sm italic ml-2 self-center">
                  ... {NUM_LAYERS - 8} more layers
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Explanation Panel */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-4">How KV Cache Works</h2>
          <div className="space-y-4 text-gray-300 leading-relaxed">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Current Step</h3>
              {tokensGenerated === 0 ? (
                <p>
                  Press "Generate Next Token" to begin. This will start the text generation process
                  and build up the KV cache.
                </p>
              ) : (
                <p>
                  Token #{tokensGenerated} ("{sampleTokens[tokensGenerated]}") has been generated.
                  The KV cache now stores K,V pairs for all {tokensGenerated + 1} tokens across
                  all {NUM_LAYERS} layers. With <strong>{currentConfig.name}</strong>, each token
                  stores {currentConfig.kvHeads} KV head(s) per layer, consuming{' '}
                  <strong>{currentStats.totalMB.toFixed(1)} MB</strong> total.
                </p>
              )}
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Why KV Cache Matters</h3>
              <p>
                During text generation, each new token must attend to <strong>all previous tokens</strong>.
                Instead of recomputing attention keys and values from scratch each time, we cache them.
                This trades memory for speed—linear speedup at the cost of growing VRAM usage.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Attention Type Impact</h3>
              <ul className="space-y-2 ml-4">
                <li>
                  <strong>MHA (Multi-Head):</strong> Full K,V cache per attention head ({NUM_HEADS} heads).
                  Most memory-intensive but highest quality.
                </li>
                <li>
                  <strong>MQA (Multi-Query):</strong> Single shared K,V head. ~{(
                    (NUM_HEADS / 1) * 100
                  ).toFixed(0)}% memory reduction, slight quality trade-off.
                </li>
                <li>
                  <strong>GQA (Grouped):</strong> K,V heads shared among groups. Best balance—~
                  {((NUM_HEADS / 8) * 100).toFixed(0)}% memory of MHA with minimal quality loss.
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Optimization Techniques</h3>
              <ul className="space-y-1 ml-4">
                <li>• <strong>PagedAttention:</strong> Allocate KV cache in blocks, reducing fragmentation.</li>
                <li>• <strong>Quantization:</strong> Store KV as int8 instead of float16 (2x memory savings).</li>
                <li>• <strong>Pruning:</strong> Drop less important tokens from cache.</li>
              </ul>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded p-3 mt-4">
              <p className="text-sm text-gray-400">
                <strong>Note:</strong> This visualization uses typical values (32 layers, 4096 hidden dim,
                32 heads). Real models vary. Actual performance depends on batch size, sequence length,
                and hardware capabilities.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
