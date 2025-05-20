---
license: mit
configs:
- config_name: 2D
  data_files:
  - split: train
    path: 2D/train_set_2D.json
  - split: test
    path: 2D/test_dataset_2D.json
- config_name: 3D
  data_files:
  - split: train
    path: 3D/train_set_3D.json
  - split: test
    path: 3D/test_dataset_3D.json
tags:
- robotics
- planning
size_categories:
- 10K<n<100K
---

# PathEval: A Benchmark for Evaluating Vision-Language Models as Evaluators for Path Planning

## Overview

Despite their promise to perform complex reasoning, large language models (LLMs) have been shown to have limited effectiveness in end-to-end planning. This has inspired an intriguing question: if these models cannot plan well, can they still contribute to the planning framework as a helpful plan evaluator? In this work, we generalize this question to consider LLMs augmented with visual understanding, i.e., Vision-Language Models (VLMs). We introduce PathEval, a novel benchmark evaluating VLMs as plan evaluators in complex path-planning scenarios. Succeeding in the benchmark requires a VLM to be able to abstract traits of optimal paths from the scenario description, demonstrate precise low-level perception on each path, and integrate this information to decide the better path. Our analysis of state-of-the-art VLMs reveals that these models face significant challenges on the benchmark. We observe that the VLMs can precisely abstract given scenarios to identify the desired traits and exhibit mixed performance in integrating the provided information. Yet, their vision component presents a critical bottleneck, with models struggling to perceive low-level details about a path. Our experimental results show that this issue cannot be trivially addressed via end-to-end fine-tuning; rather, task-specific discriminative adaptation of these vision encoders is needed for these VLMs to become effective path evaluators.

## Task Formulation

Given two paths, **P1** and **P2**, and a scenario **S**, the objective is to determine which path better satisfies the scenario's optimization criteria. Each scenario S is a high-level description that aims to optimize over a set of **path descriptors** (or **metrics**) {m1, m2, ..., mk}, where each descriptor evaluates a specific property of a path (e.g., length, smoothness, or proximity to obstacles). 

A Vision-Language Model (VLM) is presented with an image depicting P1 and P2 in the same environment side-by-side. The model must then decide which path better satisfies the scenario's criteria. To explore the sensitivity of VLMs to how a path is presented, the dataset includes both **2D and 3D images** of the path illustration. 

## Dataset Construction

### Environment Generation

An environment, as shown in Figure 1, is defined by a set of walls **O = {O₁, O₂, ..., Oₙ}**, where each wall **Oᵢ** represents an obstacle in the 2D space. Each wall is a closed geometric shape described by its vertices, and the set **O** forms the obstacles that the path must avoid. 

Our environments consist of four types of obstacle arrangements:
1. **Rings** – Environments structured as mazes with circular walls.
2. **Waves** – Consist of wavy horizontal obstacles.
3. **Mazes** – Formed by vertical and horizontal walls creating complex structures.
4. **Random** – Consist of randomly placed obstacles.

### Path Synthesis via the RRT Algorithm

To generate path candidates, we leverage the **Randomly-exploring Rapid Tree (RRT)** path planning algorithm. Starting from an initial location, the algorithm builds a tree by randomly selecting the next location while avoiding obstacles until it reaches the goal.

### Path Descriptors

For each generated path, we collect the following descriptors:

- **Minimum Clearance** – Smallest distance between any point on the path and the nearest obstacle.
- **Maximum Clearance** – Largest distance between any point on the path and the nearest obstacle.
- **Average Clearance** – Average distance between all points on the path and the nearest obstacle.
- **Path Length** – Sum of Euclidean distances between consecutive points on the path.
- **Smoothness** – Sum of the angles between consecutive segments of the path, measuring directional changes.
- **Number of Sharp Turns** – Counts the number of turns where the angle exceeds 90 degrees.
- **Maximum Angle** – Largest angle between any two consecutive segments of the path.

The three clearance metrics and path length are measured in grid size, while smoothness and maximum angle are measured in degrees. The number of sharp turns is a simple integer count. 

## Natural Language Descriptions of Scenarios

To create a sufficiently challenging path-planning benchmark, we design a total of **15 decision-making scenarios**, each optimizing different combinations of path descriptors.

## Citation

If you use **PathEval** for your work, please cite our [paper](https://arxiv.org/abs/2411.18711):

```bibtex
@inproceedings{aghzal2024evaluating,
  title={Evaluating Vision-Language Models as Evaluators in Path Planning},
  author={Aghzal, Mohamed and Yue, Xiang and Plaku, Erion and Yao, Ziyu},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```