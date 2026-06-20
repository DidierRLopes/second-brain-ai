# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Source:** https://arxiv.org/abs/2305.10601
**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (Princeton NLP)
**Published:** 2023

## Key Concepts

Extends chain-of-thought by maintaining a tree of reasoning paths rather than linear chains. Combines language model generation with search algorithms (BFS/DFS) enabling deliberate decision-making with lookahead and backtracking.

## Key Results

- 74% success on Game of 24 vs 4% for standard CoT
- Enables exploration and backtracking — can abandon bad reasoning paths
- Uses the LLM itself as both generator and evaluator of reasoning steps
- Generalizable framework applicable to many reasoning tasks
