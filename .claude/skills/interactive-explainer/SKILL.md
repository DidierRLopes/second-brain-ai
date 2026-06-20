---
name: interactive-explainer
description: >
  Build interactive, explorable visual explanations of AI/ML concepts as standalone React (.jsx) or HTML files.
  Use this skill whenever the user wants to understand a concept visually, asks for an interactive demo,
  diagram, animation, or visualization of how something works — especially concepts from their AI/ML
  knowledge base. Trigger on phrases like "explain X interactively", "visualize how Y works",
  "show me Z", "build a demo of", "interactive diagram", "animate", "explorable explanation",
  or any request to make a technical concept tangible and hands-on. Also trigger when the user
  references their knowledge base wiki topics and wants to go deeper than text.
---

# Interactive Explainer

Build rich, interactive visualizations that make AI/ML concepts tangible. These are standalone
React or HTML files that render directly in the user's environment — no build step, no server needed.

## Philosophy

The best explanations let you *play* with the concept. A slider that shows quantization precision
loss teaches more than a paragraph about it. An animated attention matrix makes self-attention
click in a way static diagrams can't. The goal is Jay Alammar-style explorable explanations
that live alongside the user's wiki articles.

## Where outputs go

Save all interactive explainers to the `interactive/` folder in the user's knowledge base:

```
ai-machine-learning/
├── raw/
├── wiki/
├── outputs/
└── interactive/    ← your output goes here
    ├── attention-mechanism.jsx
    ├── kv-cache-growth.jsx
    └── ...
```

If the `interactive/` folder doesn't exist yet, create it. Name files descriptively in kebab-case
matching the concept they explain.

## How to build an explainer

### 1. Pick the right format

**React (.jsx)** — preferred for most explainers. You get state management (useState, useReducer),
component composition, and Tailwind styling for free. Use when the explainer has interactive
controls (sliders, buttons, toggles) or animated state transitions.

**HTML (.html)** — use when you need canvas-based rendering, heavy D3 work, or WebGL via Three.js.
Everything (JS, CSS) goes in one file.

### 2. Available libraries

These are available without installation:

- **React + hooks** — `import { useState, useEffect, useMemo } from "react"`
- **Recharts** — `import { LineChart, BarChart, XAxis, YAxis, ... } from "recharts"` — great for data viz
- **D3** — `import * as d3 from "d3"` — full control over SVG-based visualizations
- **Plotly** — `import * as Plotly from "plotly"` — interactive 3D plots, good for embedding spaces
- **Three.js** — `import * as THREE from "three"` — 3D scenes (use r128 features only)
- **Lodash** — `import _ from "lodash"` — utility functions
- **MathJS** — `import * as math from "mathjs"` — matrix operations, great for showing linear algebra
- **Lucide React** — `import { Play, Pause, RotateCcw } from "lucide-react"` — icons for controls
- **Tailwind CSS** — core utility classes available, no compiler needed

### 3. Design principles

**Start with the "aha moment."** What's the one thing that, once you see it, makes the concept
click? Build around that. For attention: seeing which words attend to which. For quantization:
watching precision degrade as bits decrease. For KV cache: watching memory grow token by token.

**Interactive controls over passive animation.** Let the user drive. Sliders, step buttons,
hover states. The user should feel in control of the pace and what they explore.

**Progressive complexity.** Start simple, let the user opt into detail. A basic view that shows
the core idea, with expandable sections or tabs for the math, edge cases, or advanced variants.

**Annotate everything.** Label axes, add tooltips on hover, explain what each color means.
The visualization should be self-contained — someone should understand it without reading
the wiki article first (but link to the wiki article for deeper reading).

**Dark theme friendly.** Use `bg-gray-900 text-white` as base, with accent colors that work
on dark backgrounds. This matches most developer environments and Obsidian dark mode.

### 4. Component structure for React explainers

```jsx
import { useState, useMemo } from "react";

// Brief description comment at top
// Concept: [name]
// Wiki: [[wiki-article-name]]

export default function ConceptExplainer() {
  const [param, setParam] = useState(defaultValue);

  // Core computation/simulation logic
  const data = useMemo(() => computeData(param), [param]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      {/* Title and brief description */}
      <h1 className="text-3xl font-bold mb-2">Concept Name</h1>
      <p className="text-gray-400 mb-8">One-line description of what you're seeing.</p>

      {/* Main visualization area */}
      <div className="...">
        {/* The core visual */}
      </div>

      {/* Interactive controls */}
      <div className="mt-6 ...">
        {/* Sliders, buttons, toggles */}
      </div>

      {/* Explanation panel */}
      <div className="mt-6 text-gray-300 text-sm">
        {/* What's happening, keyed to current state */}
      </div>
    </div>
  );
}
```

### 5. Quality checklist

Before saving, verify:
- [ ] Default state shows something meaningful (not blank)
- [ ] All controls work and update the visualization
- [ ] Labels and annotations are present
- [ ] Color choices are accessible (not just red/green)
- [ ] No localStorage or sessionStorage usage (not supported)
- [ ] Component has a default export with no required props
- [ ] Explanation text updates based on the current state of the controls

## Connecting to the knowledge base

When building an explainer, check the user's `wiki/` folder for the relevant article. Reference
it in a comment at the top of the file. If the explainer reveals something the wiki article
doesn't cover, mention it to the user so the wiki can be updated — this is the compounding
loop that makes the knowledge base smarter over time.

## Example prompts this skill handles

- "Explain attention interactively"
- "Build me a visualization of how quantization works"
- "I want to see how the KV cache grows during generation"
- "Show me how LoRA low-rank decomposition works"
- "Make an interactive demo of embeddings in 2D space"
- "Visualize the transformer architecture"
- "How does speculative decoding work? Show me"

<!--
NOTE TO FUTURE AGENTS: This skill should evolve as the knowledge base grows.
When creating new explainers, if you notice patterns or reusable components
(e.g., a matrix visualization widget, a common slider control panel, a token
sequence renderer), consider extracting them into a shared references/ file
that future explainers can import patterns from. The interactive/ folder and
this skill are living artifacts — update them as the knowledge base expands
with new topics and as you discover better visualization approaches.
-->
