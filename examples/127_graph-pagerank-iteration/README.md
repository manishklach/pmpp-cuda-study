# 127 - Graph Pagerank Iteration

## Overview

Run one or more PageRank iterations over a small graph.

## Why this matters

PageRank is a simple graph kernel with practical gather-style structure.

## Expected kernel structure

Map outgoing or incoming edges, accumulate contributions, and apply damping after each iteration.

## Future implementation notes

Explain the dataflow clearly before optimizing storage layout.

## Build and run

```powershell
nvcc -std=c++17 main.cu -o example.exe
.\example.exe
```
