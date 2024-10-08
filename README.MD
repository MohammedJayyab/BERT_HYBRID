# Hybrid Recommendation System Using BERT and Collaborative Filtering

## Overview

This project implements an advanced **hybrid recommendation system** that combines **content-based filtering** (using BERT embeddings) with **collaborative filtering** (using user-product interactions) to generate personalized product recommendations. The system incorporates **multi-hop traversal** in a user-product interaction graph to explore both direct and indirect relationships between products and users, offering a highly tailored recommendation experience.

The system blends **content similarity** derived from product descriptions with **collaborative similarities** based on user interactions, such as clicks, additions to the cart, or purchases. The final hybrid model merges these similarities to provide accurate, contextually relevant recommendations.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data Requirements](#data-requirements)
3. [Algorithm Design](#algorithm-design)
   - [Content-Based Filtering](#content-based-filtering)
   - [Collaborative Filtering](#collaborative-filtering)
   - [Hybrid Model](#hybrid-model)
4. [Implementation Details](#implementation-details)
   - [1. Preparing Product Documents](#1-preparing-product-documents)
   - [2. Generating BERT Embeddings](#2-generating-bert-embeddings)
   - [3. Constructing the User-Product Graph](#3-constructing-the-user-product-graph)
   - [4. Normalizing Edge Weights](#4-normalizing-edge-weights)
   - [5. Computing Combined Similarities](#5-computing-combined-similarities)
5. [Multi-Hop Recommendations](#multi-hop-recommendations)
   - [1-Hop Traversal](#1-hop-traversal)
   - [2-Hop Traversal](#2-hop-traversal)
6. [How to Use](#how-to-use)
7. [Example Output](#example-output)
8. [Dependencies](#dependencies)
9. [Future Enhancements](#future-enhancements)

## Project Structure

The `HybridRecommendationSystem` class implements the recommendation system, which includes:

- **Data Loading**: Reads product and transaction data.
- **Content-Based Filtering**: Uses BERT embeddings to generate semantic similarity scores between products.
- **Collaborative Filtering**: Uses a graph-based approach to model user-product interactions.
- **Hybrid Recommendations**: Combines content-based and collaborative filtering.
- **Multi-Hop Graph Traversal**: Generates recommendations using both direct and indirect relationships between users and products.

## Data Requirements

### 1. `products.csv`

This file contains information about products, with columns:

- **`product_id`**: Unique identifier for the product.
- **`product_name`**: Name of the product.
- **`category_name`**: Category to which the product belongs.
- **`description`**: Description of the product for generating embeddings.

### 2. `transactions.csv`

This file logs user interactions with products, with columns:

- **`customer_id`**: Unique identifier for the user.
- **`product_id`**: Identifier for the product interacted with.
- **`interaction_type`**: Type of interaction (e.g., 'clicked', 'added', 'purchased').
- **`quantity`**: Number of units interacted with (default is 1).

## Algorithm Design

### Content-Based Filtering

Content-based filtering involves generating product embeddings using **BERT**. Each product's description, name, and category are encoded into a vector representation, allowing us to calculate the **cosine similarity** between products.

**Mathematical Formula**:

- Let ![E_i](https://latex.codecogs.com/svg.latex?E_i) be the embedding of product \( i \). be the embedding of product \( i \).

- **Cosine Similarity** between product \( i \) and product \( j \) is calculated as:

![Equation](<https://latex.codecogs.com/svg.latex?\text{Similarity}(i,j)=\frac{E_i\cdot{E_j}}{||E_i||\cdot{||E_j||}}>)

**Python Code Example**:

```python
import numpy as np

def cosine_similarity(embedding_i, embedding_j):
    # Compute the dot product of the two embeddings
    dot_product = np.dot(embedding_i, embedding_j)
    # Compute the L2 norm (magnitude) of each embedding
    norm_i = np.linalg.norm(embedding_i)
    norm_j = np.linalg.norm(embedding_j)
    # Calculate cosine similarity
    similarity = dot_product / (norm_i * norm_j)
    return similarity

# Example usage
embedding_i = np.array([0.4, 0.6, 0.8])
embedding_j = np.array([0.3, 0.7, 0.9])
cosine_sim = cosine_similarity(embedding_i, embedding_j)
print(f"Cosine Similarity: {cosine_sim:.3f}")
```

**Example Calculation**:
Suppose we have two product embeddings:

- \( E_A = [0.4, 0.6, 0.8] \)
- \( E_B = [0.3, 0.7, 0.9] \)

The cosine similarity is calculated as follows:
\[
\text{Similarity}(A, B) = \frac{(0.4 \times 0.3) + (0.6 \times 0.7) + (0.8 \times 0.9)}{\sqrt{0.4^2 + 0.6^2 + 0.8^2} \times \sqrt{0.3^2 + 0.7^2 + 0.9^2}} = 0.973
\]

### Collaborative Filtering

Collaborative filtering is implemented using a **user-product graph** where:

- **Nodes** represent users and products.
- **Edges** represent interactions, with weights based on interaction types and quantities.

**Edge Weight Calculation**:

- Let \( W\_{\text{base}} \) be the base weight for an interaction (`clicked`: 1, `added`: 2, `purchased`: 3).
- Let \( Q \) be the quantity of the product interacted with.
- The final weight \( W\_{\text{final}} \) is:

  \[
  W*{\text{final}} = W*{\text{base}} \times \log(Q + 1)
  \]

**Example Graph Representation**:

| User     | Product   | Interaction Type | Quantity | Weight (\( W\_{\text{final}} \)) |
| -------- | --------- | ---------------- | -------- | -------------------------------- |
| User 101 | Product A | Clicked          | 1        | 0.69                             |
| User 101 | Product B | Purchased        | 2        | 2.08                             |
| User 102 | Product B | Clicked          | 1        | 0.69                             |
| User 102 | Product C | Added to Cart    | 3        | 2.20                             |

### Normalization of Similarities

To ensure consistency, the following normalization techniques are used:

1. **Min-Max Normalization** for the similarity matrix:

   \[
   S*{\text{normalized}} = \frac{S - S*{\text{min}}}{S*{\text{max}} - S*{\text{min}}}
   \]

   Where \( S \) is the similarity score, \( S*{\text{min}} \) and \( S*{\text{max}} \) are the minimum and maximum values in the similarity matrix.

2. **Z-score Normalization** for collaborative similarities:

   \[
   S\_{\text{zscore}} = \frac{S - \mu}{\sigma}
   \]

   Where \( \mu \) is the mean and \( \sigma \) is the standard deviation.

### Hybrid Model

The final recommendation combines content-based similarity (\( S*{\text{content}} \)) and collaborative similarity (\( S*{\text{collab}} \)):

\[
S*{\text{combined}} = \alpha \times S*{\text{content}} + \beta \times S\_{\text{collab}}
\]

Where \( \alpha \) and \( \beta \) are hyperparameters that determine the contribution of each similarity type.

### Custom Scoring Function

**Scoring Formula** for multi-hop recommendations:

\[
S = (W*{\text{interaction}} + S*{\text{content}}) \times \log(Q + 1)
\]

Where:

- \( W\_{\text{interaction}} \): The interaction weight, based on interaction type and quantity.
- \( S\_{\text{content}} \): The content-based similarity of the product.
- \( Q \): The quantity of the product.

**Example Calculation**:

- **Interaction**: User purchased 2 units of **Product B**.
- **Interaction Weight**: \( W\_{\text{interaction}} = 3 \times 2 = 6 \)
- **Content Similarity**: \( S\_{\text{content}} = 0.8 \)
- **Quantity (\( Q \))**: \( Q = 2 \)
- **Score**:

  \[
  S = (6 + 0.8) \times \log(2 + 1) = 6.8 \times 1.1 = 7.48
  \]

## Implementation Details

### 1. Preparing Product Documents

The `prepare_product_documents` method creates a text representation for each product by concatenating its name, category, and description.

### 2. Generating BERT Embeddings

The `generate_bert_embeddings` method uses BERT to generate vector embeddings for each product document. These embeddings are crucial for capturing semantic meaning and calculating content similarity.

### 3. Constructing the User-Product Graph

The `construct_user_product_graph` method creates a graph with users and products as nodes, and interactions as weighted edges.

**Graph Example**:

- **Nodes**: Users (`user_101`, `user_102`) and Products (`product_A`, `product_B`, `product_C`).
- **Edges**: Represent interactions with different weights (e.g., click, add to cart, purchase).

### 4. Normalizing Edge Weights

The `normalize_edge_weights` method normalizes edge weights using Min-Max normalization to ensure all weights fall between 0 and 1.

### 5. Computing Combined Similarities

The `compute_combined_similarity` method calculates the final similarity score between products by combining content-based and collaborative similarities using the formula:

\[
S*{\text{combined}} = \alpha \times S*{\text{content}} + \beta \times S\_{\text{collab}}
\]

## Multi-Hop Recommendations

### 1-Hop Traversal

A **1-hop traversal** finds products directly interacted with by a user.

**Example**:

- **User 101** has interacted with **Product A** and **Product B**.
- **1-Hop Recommendation**: Recommend **Product A** and **Product B**.

### 2-Hop Traversal

A **2-hop traversal** finds products through another user, increasing diversity in recommendations.

**Example**:

- **User 101** has interacted with **Product A**.
- **User 102** has interacted with **Product A** and **Product C**.
- **2-Hop Recommendation for User 101**: Recommend **Product C** since **User 102** also interacted with **Product A**.

| User     | Product Interactions |
| -------- | -------------------- |
| User 101 | Product A            |
| User 102 | Product A, Product C |

**Recommendation**: Recommend **Product C** to **User 101**.

## How to Use

```python
recommender = HybridRecommendationSystem('data/products.csv', 'data/transactions.csv')
recommender.run_recommendation_pipeline()
```

## Example Output

- **Top 5 Recommendations for User 101**: Based on the multi-hop traversal, specific product recommendations are generated.

## Dependencies

- `transformers`
- `torch`
- `pandas`
- `networkx`
- `sklearn`

## Future Enhancements

- **Incorporate Temporal Dynamics**: Consider the time of interactions to capture trends.
- **Advanced Weighting Mechanisms**: Use machine learning models to optimize interaction weights.

This completes the README for the Hybrid E-Commerce Recommendation System, providing detailed explanations, equations, and illustrative examples for better understanding.
