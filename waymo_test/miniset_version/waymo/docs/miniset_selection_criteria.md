# Waymo Miniset Selection Criteria

This document summarizes the criteria used to construct the Waymo validation miniset. The goal is to provide a compact benchmark that is diverse, challenging, and suitable for analyzing spatial-temporal reasoning in driving scenes.

Final dataset:

```text
dataset/waymo_validation_balanced_diverse_70hard_option_balanced_miniset_from_full_results.json
```

## Overview

The miniset contains 2,400 questions, with 100 examples for each of 24 question types:

```text
SC-1, SC-2, SC-3, SC-4,
SP-2, SP-3, SP-3a,
TE-1, TE-2, TE-4, TE-5,
TM-2, TM-3, TM-5,
TRJ-1, TRJ-2, TRJ-3, TRJ-4, TRJ-5, TRJ-6, TRJ-7, TRJ-8, TRJ-9, TRJ-10
```

The question types cover scene understanding, spatial perception, temporal extrapolation, temporal memory, and trajectory reasoning.

## Selection Goals

The miniset was not selected by random sampling alone, nor was it selected only from model failure cases. Instead, we used a curated selection process with the following goals.

## 1. Balanced Task Coverage

Each question type contributes the same number of examples.

This avoids letting frequent or easier task categories dominate the benchmark. It also makes per-task and cross-task comparisons clearer, since each capability area has the same evaluation scale.

## 2. Scene Diversity

The selected examples are encouraged to come from many different Waymo scenes and temporal windows.

This reduces redundancy. If many questions come from the same scene or nearby frames, model performance may reflect repeated visual context rather than general reasoning ability. Scene diversity makes the miniset more representative and more useful for evaluating generalization across driving environments.

## 3. Scenario Diversity

The selection process considers the distribution of driving scenarios, such as intersections, turns, lane changes, merges, cut-ins, pedestrians, cyclists, dense traffic, construction zones, and simpler residential scenes.

This is important because spatial-temporal reasoning is highly scenario-dependent. A model may perform well on straight driving but fail at intersections or during multi-agent interactions. Including diverse scenarios makes the benchmark better at exposing these differences.

## 4. Ground-Truth Diversity

We also consider the diversity of ground-truth answers and labels.

For multiple-choice questions, this helps avoid repeated answer patterns. For numeric, free-response, and trajectory-output questions, it helps cover different distances, motion patterns, risk sources, and explanation types. This reduces shortcut behavior and makes the benchmark less dependent on superficial answer priors.

## 5. Difficulty

A reference model was run on the full validation question set, and its responses were used as a difficulty signal.

The miniset includes many examples that are difficult for the reference model, but it also keeps some correctly answered examples. This prevents the dataset from becoming a model-specific failure set. The resulting benchmark is challenging while still supporting meaningful analysis of both success and failure cases.

## 6. Complex Driving Cases

The selection favors examples involving richer driving interactions, such as turns, merges, cut-ins, intersections, crosswalks, vulnerable road users, occlusions, dense traffic, and situations where the ego vehicle may need to slow down, yield, or change its plan.

These cases are valuable because they require more than static object recognition. They test whether a model can reason about motion, intent, interaction, risk, and future behavior.

## 7. Simple Scene Control

Simple scenes, such as stable straight driving with no significant interaction, are not removed entirely. However, their proportion is controlled.

Keeping some simple examples is useful for measuring basic capability. Controlling their frequency ensures that the miniset remains challenging and does not overestimate model performance through many low-information cases.

## 8. Human Review

Human review records are used for scene-level and trajectory-related questions when available.

This improves label quality for cases where automatic generation may be ambiguous, especially for risk descriptions, preferred actions, trajectory choices, and free-form explanations. Human review makes the benchmark more reliable for paper-level analysis.

## 9. MCQ Option Balancing

After selecting the question rows, the answer choices for multiple-choice questions are permuted to balance the correct option letters.

This does not change the questions or their semantic answers. It only reduces answer-position bias, so models cannot gain an artificial advantage by favoring a particular option letter.

## Interpretation

The Waymo miniset should be interpreted as a curated stress-test subset rather than an unbiased random sample of the full Waymo validation split.

This design is useful for evaluating spatial-temporal reasoning because it balances coverage, diversity, difficulty, and label reliability. It is intended to support detailed model comparison and failure analysis, especially in complex driving situations where perception, memory, prediction, and decision-relevant reasoning must work together.
