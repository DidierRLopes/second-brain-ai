// Concept: Embeddings | Wiki: [[embeddings]] [[retrieval-augmented-generation]]

import React, { useState, useMemo } from 'react';

// Pre-defined word embeddings in 2D space with semantic clusters
const WORDS_DATA = [
  // Animals cluster
  { word: 'cat', x: 15, y: 85, cluster: 'Animals' },
  { word: 'dog', x: 25, y: 80, cluster: 'Animals' },
  { word: 'fish', x: 18, y: 92, cluster: 'Animals' },
  { word: 'bird', x: 22, y: 75, cluster: 'Animals' },
  { word: 'horse', x: 12, y: 78, cluster: 'Animals' },

  // Colors cluster
  { word: 'red', x: 72, y: 85, cluster: 'Colors' },
  { word: 'blue', x: 78, y: 90, cluster: 'Colors' },
  { word: 'green', x: 75, y: 78, cluster: 'Colors' },
  { word: 'yellow', x: 82, y: 88, cluster: 'Colors' },
  { word: 'purple', x: 70, y: 82, cluster: 'Colors' },

  // Countries cluster
  { word: 'France', x: 45, y: 20, cluster: 'Countries' },
  { word: 'Germany', x: 52, y: 25, cluster: 'Countries' },
  { word: 'Japan', x: 48, y: 15, cluster: 'Countries' },
  { word: 'Brazil', x: 55, y: 22, cluster: 'Countries' },
  { word: 'Canada', x: 42, y: 28, cluster: 'Countries' },

  // Tech cluster
  { word: 'computer', x: 25, y: 45, cluster: 'Tech' },
  { word: 'algorithm', x: 32, y: 48, cluster: 'Tech' },
  { word: 'neural', x: 28, y: 52, cluster: 'Tech' },
  { word: 'tensor', x: 35, y: 42, cluster: 'Tech' },
  { word: 'matrix', x: 22, y: 50, cluster: 'Tech' },

  // Food cluster
  { word: 'pizza', x: 68, y: 35, cluster: 'Food' },
  { word: 'sushi', x: 75, y: 40, cluster: 'Food' },
  { word: 'pasta', x: 70, y: 32, cluster: 'Food' },
  { word: 'burger', x: 78, y: 38, cluster: 'Food' },
  { word: 'salad', x: 72, y: 45, cluster: 'Food' },

  // Additional words for richer exploration
  { word: 'tiger', x: 12, y: 88, cluster: 'Animals' },
  { word: 'elephant', x: 28, y: 85, cluster: 'Animals' },
  { word: 'orange', x: 75, y: 72, cluster: 'Colors' },
  { word: 'Spain', x: 50, y: 18, cluster: 'Countries' },
  { word: 'python', x: 38, y: 55, cluster: 'Tech' },
  { word: 'rice', x: 68, y: 48, cluster: 'Food' },
];

const CLUSTER_COLORS = {
  Animals: '#ffffff',
  Colors: '#cccccc',
  Countries: '#999999',
  Tech: '#666666',
  Food: '#333333',
};

const CLUSTER_RADIUS = {
  Animals: 4,
  Colors: 5,
  Countries: 6,
  Tech: 5,
  Food: 4,
};

// Calculate distance between two points
const calculateDistance = (p1, p2, metric = 'euclidean') => {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;

  if (metric === 'euclidean') {
    return Math.sqrt(dx * dx + dy * dy);
  } else {
    // Cosine similarity approximation: convert distances to similarity (higher = more similar)
    const dist = Math.sqrt(dx * dx + dy * dy);
    const maxDist = 100;
    return Math.max(0, 1 - dist / maxDist);
  }
};

// Calculate K nearest neighbors
const getKNearestNeighbors = (word, k = 3, metric = 'euclidean') => {
  const wordData = WORDS_DATA.find((w) => w.word === word);
  if (!wordData) return [];

  const distances = WORDS_DATA.filter((w) => w.word !== word).map((w) => ({
    ...w,
    distance: calculateDistance(wordData, w, metric),
  }));

  if (metric === 'euclidean') {
    distances.sort((a, b) => a.distance - b.distance);
  } else {
    distances.sort((a, b) => b.distance - a.distance);
  }

  return distances.slice(0, k);
};

export default function EmbeddingSpaceExplorer() {
  const [selectedWord, setSelectedWord] = useState('cat');
  const [k, setK] = useState(3);
  const [searchTerm, setSearchTerm] = useState('');
  const [metric, setMetric] = useState('euclidean');
  const [showVectorArithmetic, setShowVectorArithmetic] = useState(false);

  // Filter words by search term
  const filteredWords = useMemo(
    () =>
      WORDS_DATA.filter((w) =>
        w.word.toLowerCase().includes(searchTerm.toLowerCase())
      ),
    [searchTerm]
  );

  // Get nearest neighbors for selected word
  const neighbors = useMemo(
    () => getKNearestNeighbors(selectedWord, k, metric),
    [selectedWord, k, metric]
  );

  const selectedWordData = WORDS_DATA.find((w) => w.word === selectedWord);

  // Vector arithmetic demo: King - Man + Woman
  const kingData = WORDS_DATA.find((w) => w.word === 'horse');
  const manData = WORDS_DATA.find((w) => w.word === 'dog');
  const womanData = WORDS_DATA.find((w) => w.word === 'bird');

  const resultVector = useMemo(() => {
    if (!kingData || !manData || !womanData) return null;
    return {
      x: kingData.x - manData.x + womanData.x,
      y: kingData.y - manData.y + womanData.y,
    };
  }, []);

  return (
    <div className="bg-black text-white p-8 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-2">Embedding Space Explorer</h1>
        <p className="text-gray-500 mb-6 text-sm">
          Visualize how embeddings encode semantic meaning as geometry in 2D space
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Main Plot */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
              <svg
                width="100%"
                height={500}
                viewBox="0 0 500 500"
                className="border border-gray-800 rounded bg-black"
              >
                {/* Grid background */}
                <defs>
                  <pattern
                    id="grid"
                    width="50"
                    height="50"
                    patternUnits="userSpaceOnUse"
                  >
                    <path
                      d="M 50 0 L 0 0 0 50"
                      fill="none"
                      stroke="#333333"
                      strokeWidth="0.5"
                    />
                  </pattern>
                </defs>
                <rect width="500" height="500" fill="url(#grid)" />

                {/* Draw connections to nearest neighbors */}
                {selectedWordData &&
                  neighbors.map((neighbor, idx) => (
                    <g key={`connection-${neighbor.word}`}>
                      <line
                        x1={selectedWordData.x * 5}
                        y1={selectedWordData.y * 5}
                        x2={neighbor.x * 5}
                        y2={neighbor.y * 5}
                        stroke="#ffffff"
                        strokeWidth="1"
                        opacity="0.4"
                        strokeDasharray="3,3"
                      />
                      {/* Show distance/similarity score */}
                      <text
                        x={(selectedWordData.x + neighbor.x) * 2.5}
                        y={(selectedWordData.y + neighbor.y) * 2.5}
                        fontSize="10"
                        fill="#999999"
                        textAnchor="middle"
                        className="pointer-events-none"
                      >
                        {metric === 'euclidean'
                          ? neighbor.distance.toFixed(1)
                          : neighbor.distance.toFixed(3)}
                      </text>
                    </g>
                  ))}

                {/* Draw vector arithmetic arrows if enabled */}
                {showVectorArithmetic && kingData && manData && womanData && resultVector && (
                  <g opacity="0.5">
                    {/* Horse vector (solid white) */}
                    <line
                      x1="0"
                      y1="0"
                      x2={kingData.x * 5}
                      y2={kingData.y * 5}
                      stroke="#ffffff"
                      strokeWidth="2"
                    />
                    {/* Dog vector (negative, light gray dashed) */}
                    <line
                      x1={kingData.x * 5}
                      y1={kingData.y * 5}
                      x2={(kingData.x - manData.x) * 5}
                      y2={(kingData.y - manData.y) * 5}
                      stroke="#cccccc"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                    />
                    {/* Bird vector (positive, medium gray) */}
                    <line
                      x1={(kingData.x - manData.x) * 5}
                      y1={(kingData.y - manData.y) * 5}
                      x2={resultVector.x * 5}
                      y2={resultVector.y * 5}
                      stroke="#999999"
                      strokeWidth="2"
                    />
                    {/* Result point */}
                    <circle
                      cx={resultVector.x * 5}
                      cy={resultVector.y * 5}
                      r="5"
                      fill="#ffffff"
                      stroke="#666666"
                      strokeWidth="2"
                    />
                  </g>
                )}

                {/* Draw all words */}
                {WORDS_DATA.map((wordItem) => {
                  const isSelected = wordItem.word === selectedWord;
                  const isHighlighted = filteredWords.some(
                    (w) => w.word === wordItem.word
                  );
                  const isNeighbor = neighbors.some(
                    (n) => n.word === wordItem.word
                  );
                  const baseRadius = CLUSTER_RADIUS[wordItem.cluster];

                  return (
                    <g key={wordItem.word}>
                      <circle
                        cx={wordItem.x * 5}
                        cy={wordItem.y * 5}
                        r={isSelected ? baseRadius + 3 : isNeighbor ? baseRadius + 2 : baseRadius}
                        fill={CLUSTER_COLORS[wordItem.cluster]}
                        stroke={isSelected ? '#ffffff' : 'none'}
                        strokeWidth={isSelected ? '2' : '0'}
                        opacity={
                          isHighlighted || searchTerm === '' ? 1 : 0.3
                        }
                        className="cursor-pointer transition-all"
                        onClick={() => setSelectedWord(wordItem.word)}
                        onMouseEnter={(e) => {
                          e.target.setAttribute('r', isSelected ? baseRadius + 3 : baseRadius + 2);
                          e.target.setAttribute('stroke-width', '1');
                          e.target.setAttribute('stroke', '#ffffff');
                        }}
                        onMouseLeave={(e) => {
                          e.target.setAttribute(
                            'r',
                            isSelected ? baseRadius + 3 : isNeighbor ? baseRadius + 2 : baseRadius
                          );
                          e.target.setAttribute(
                            'stroke-width',
                            isSelected ? '2' : '0'
                          );
                        }}
                      >
                        <title>{`${wordItem.word} (${wordItem.cluster})`}</title>
                      </circle>
                      <text
                        x={wordItem.x * 5}
                        y={wordItem.y * 5 + 15}
                        fontSize="10"
                        fill="white"
                        textAnchor="middle"
                        className="pointer-events-none select-none"
                        opacity={
                          isHighlighted || searchTerm === '' ? 1 : 0.3
                        }
                      >
                        {wordItem.word}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Cluster Legend */}
            <div className="grid grid-cols-5 gap-2 mt-4">
              {Object.entries(CLUSTER_COLORS).map(([cluster, color]) => (
                <div key={cluster} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded-full border border-gray-600"
                    style={{ backgroundColor: color }}
                  ></div>
                  <span className="text-sm text-gray-400">{cluster}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Control Panel */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 h-fit">
            <h2 className="text-xl font-bold mb-4">Controls</h2>

            {/* Selected Word Info */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-400 mb-2">
                Selected Word
              </label>
              <div className="bg-black rounded p-3 border border-gray-800">
                <p className="text-lg font-bold text-white">
                  {selectedWord}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Cluster:{' '}
                  <span
                    style={{
                      color: CLUSTER_COLORS[selectedWordData?.cluster],
                    }}
                  >
                    {selectedWordData?.cluster}
                  </span>
                </p>
                <p className="text-xs text-gray-500">
                  Coords: ({selectedWordData?.x.toFixed(1)},{' '}
                  {selectedWordData?.y.toFixed(1)})
                </p>
              </div>
            </div>

            {/* K Slider */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-400 mb-2">
                Nearest Neighbors (K={k})
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer"
              />
              <div className="mt-3 space-y-1">
                {neighbors.map((neighbor, idx) => (
                  <div
                    key={neighbor.word}
                    className="text-xs text-gray-400 flex justify-between"
                  >
                    <span>{neighbor.word}</span>
                    <span className="text-gray-600">
                      {metric === 'euclidean'
                        ? `${neighbor.distance.toFixed(1)} u`
                        : `${neighbor.distance.toFixed(3)} sim`}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Metric Toggle */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-400 mb-2">
                Similarity Metric
              </label>
              <div className="flex gap-2">
                {['euclidean', 'cosine'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-colors ${
                      metric === m
                        ? 'bg-white text-black'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    {m === 'euclidean' ? 'Euclidean' : 'Cosine'}
                  </button>
                ))}
              </div>
            </div>

            {/* Search Filter */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-400 mb-2">
                Search/Filter
              </label>
              <input
                type="text"
                placeholder="e.g., 'dog', 'blue'..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 bg-black border border-gray-800 rounded text-sm text-white placeholder-gray-700 focus:outline-none focus:border-gray-400"
              />
              {searchTerm && (
                <p className="text-xs text-gray-500 mt-2">
                  Found: {filteredWords.length} match
                  {filteredWords.length !== 1 ? 'es' : ''}
                </p>
              )}
            </div>

            {/* Vector Arithmetic Demo */}
            <div className="mb-4">
              <button
                onClick={() => setShowVectorArithmetic(!showVectorArithmetic)}
                className={`w-full px-4 py-2 rounded font-medium text-sm transition-colors ${
                  showVectorArithmetic
                    ? 'bg-white text-black'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {showVectorArithmetic ? '✓' : '◇'} Vector Arithmetic Demo
              </button>
            </div>

            {showVectorArithmetic && (
              <div className="bg-black rounded p-3 border border-gray-800 text-xs text-gray-400 space-y-1">
                <p>
                  <span className="text-white">horse</span> −{' '}
                  <span className="text-white">dog</span> +{' '}
                  <span className="text-white">bird</span> = ?
                </p>
                <p className="text-gray-600">
                  Shows vector arithmetic on the plot (white arrows with varying shades)
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Explanation Section */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <h2 className="text-2xl font-bold mb-4">How Embeddings Work</h2>
          <div className="space-y-4 text-gray-400">
            <p>
              <strong>Embeddings</strong> are numerical representations of words,
              concepts, or images in a continuous vector space. In this explorer,
              each word is represented as a point in 2D space.
            </p>
            <p>
              <strong>Semantic Clustering:</strong> Notice how related words cluster
              together. Animals group together, colors group together, and so on. This
              happens because words with similar meanings have similar embeddings—geometry
              encodes meaning.
            </p>
            <p>
              <strong>Similarity Metrics:</strong> Two common ways to measure distance
              between embeddings are:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-2">
              <li>
                <strong>Euclidean Distance:</strong> The straight-line distance between
                two points. Smaller values mean closer/more similar points.
              </li>
              <li>
                <strong>Cosine Similarity:</strong> Measures the angle between vectors
                (0 = different direction, 1 = same direction). More robust to magnitude.
              </li>
            </ul>
            <p>
              <strong>Vector Arithmetic:</strong> Embeddings support algebraic operations.
              The classic example is: King - Man + Woman ≈ Queen. This works because
              the vector differences encode relationships like gender and status.
            </p>
            <p>
              <strong>Applications:</strong> Embeddings power modern AI: semantic search,
              recommendation systems, language models, image similarity, and more. They're
              the foundation of RAG (Retrieval-Augmented Generation).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
