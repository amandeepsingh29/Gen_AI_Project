# Concept-Guided Skin Cancer Prediction with LLM-Powered Clinical Explanations

This project implements an interpretable AI assistant for dermoscopic skin lesion analysis.  
Instead of predicting a diagnosis directly from an image, the system first predicts **clinically meaningful dermoscopic concepts** and then uses them to form the final diagnosis. An LLM layer generates **clear, clinician-friendly explanations** of the model's reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Concept Bottleneck Model (CBM)](#concept-bottleneck-model-cbm)
  - [Training Strategy](#training-strategy)
  - [LLM-Based Explanation Layer](#llm-based-explanation-layer)
  - [Explainability Tools](#explainability-tools)
- [Results](#results)
- [User Interface](#user-interface)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Authors](#authors)
- [Disclaimer](#disclaimer)

---

## Overview

Dermatologists rely on dermoscopic concepts such as:

- Atypical pigment network  
- Blue-whitish veil  
- Atypical dots and globules  
- Irregular streaks  
- Irregular pigmentation  
- Atypical vascular structures  
- Regression structures  

Most deep learning systems jump straight from **image → diagnosis** and never expose this intermediate reasoning, which makes them hard to trust and verify in clinical settings.  

This project bridges that gap by:

1. Predicting the seven dermoscopic concepts first.  
2. Using those predicted concepts (and only those) to form the final diagnosis.  
3. Generating visual and textual explanations that match clinical thinking.  

---

## Key Features

- **Concept Bottleneck Model (CBM)**  
  - Image → 7 dermoscopic concept probabilities → final diagnosis.  

- **Interpretable Diagnosis Predictor**  
  - Small feed-forward network that takes only concept scores as input.

- **Explainable AI Toolkit**
  - Concept Activation Maps (CAMs)
  - Region-of-Interest (ROI) masking
  - Concept influence scores
  - Natural-language reasoning via LLM  

- **Interactive UI**
  - Upload image, view predictions, heatmaps, influence plots, and chat with an LLM ("Dr. Bot").

---

## Dataset

**Dataset:** Derm7pt (dermoscopic images with concept annotations).  

- ~1500 dermoscopic images of benign, atypical, and malignant lesions.  
- For each image:
  - Diagnosis label
  - Seven dermoscopic concept labels  
- Well-suited to concept-based modeling (concept prediction + diagnosis).  

### Preprocessing

- Images resized to **224 × 224** pixels.  
- Pixel values normalized.  
- Data augmentation: flips, rotations, color jitter.  
- Data split:
  - 80% training, 20% testing  
  - Separate validation set  
  - **Patient-level separation** to avoid data leakage  
  - Fixed random seed for reproducibility  

---

## Methodology

The system contains three main components:

1. **Concept Bottleneck Model (CBM)**
2. **Diagnosis Predictor**
3. **LLM-Based Clinical Explanation Layer**

### Concept Bottleneck Model (CBM)

#### Concept Predictor

- Backbone: **ResNet-18**.  
- Output: 7 independent concept probabilities.  
- Activation: **Sigmoid** (multi-label).  
- Loss: **Binary Cross-Entropy (BCE)**.  
- Produces a **7-dimensional concept vector** representing the lesion in clinical terms.  

#### Diagnosis Predictor

- Input: 7-D concept vector (only).  
- Architecture: small fully-connected network with hidden layer + dropout.  
- Output: class probabilities via **softmax**.  
- Loss: **Cross-Entropy**.  

### Training Strategy

Two-stage training to keep the reasoning interpretable:

1. **Stage 1 – Concept Predictor**
   - Train only the Concept Predictor to recognize the seven dermoscopic concepts from images.
   - Diagnosis Predictor is frozen.

2. **Stage 2 – Diagnosis Predictor**
   - Freeze Concept Predictor weights.
   - Train Diagnosis Predictor using the fixed concept probabilities as inputs.

This ensures the final diagnosis is strictly based on interpretable concepts.

### LLM-Based Explanation Layer

- Input to LLM:
  - Predicted concepts + their probabilities
  - Concept influence scores
  - High-level info from heatmaps / highlighted regions  
- Output:
  - A short, clinically oriented explanation that connects
    concepts → evidence → diagnosis.  
- Important:
  - LLM **does not modify** the prediction, it only explains it.  

### Explainability Tools

1. **Concept Activation Maps (CAMs)**
   - Separate heatmap for each dermoscopic concept.  

2. **Region-of-Interest (ROI) Masking**
   - Non-informative regions are dimmed/greyed out to make key regions stand out.  

3. **Concept Influence Scores**
   - Remove one concept at a time and measure the change in prediction.
   - Provides a ranked list of which concepts mattered most.  

4. **LLM-Generated Clinical Reasoning**
   - Turns probabilities and visuals into a coherent narrative explanation.  

---

## Results

The model achieves strong overall performance:

- **Accuracy:** ~81%  
- **Precision:** ~84%  
- **Recall:** ~81%  
- **F1-score:** ~81%  

These results show that adding interpretability through a CBM **does not significantly reduce** predictive performance.  

### Class-wise Performance

Better performance is observed on visually distinctive classes, such as:

- Dermatofibroma  
- Melanosis  
- Recurrent nevus  
- Several melanoma subtypes  

### Explainability Outcomes

For each prediction, the system can provide:

- Concept-specific heatmaps  
- ROI views  
- Influence score plots  
- Natural-language reasoning  

Together these form a full explanation pipeline.

---

## User Interface

The UI is designed for clinicians, not ML engineers. It allows users to:

- Upload a dermoscopic image.  
- Choose the prediction model.  
- View:
  - Diagnosis
  - Concept scores
  - Heatmaps & ROI masks
  - Influence scores
- Interact with an LLM chat panel (“Dr. Bot”) to:
  - Get a written explanation.
  - Ask follow-up questions about concepts or evidence.

---
