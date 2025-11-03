# 🧠 Explainability for Vision Transformers (ViT)

This project explores *explainability techniques for Vision Transformers*, especially:
- **Attention Flow**
- **Attention Rollout**

These methods help visualize how information flows through transformer layers and which image regions contribute most to a prediction.

---

## 📌 Objective

Vision Transformers achieve strong performance, but their decisions are often hard to interpret.  
This project implements and compares explainability methods to better understand the attention mechanisms within ViTs.

---

## ⚙️ Methods

### ✅ 1. **Attention Rollout**  
- Aggregates attention matrices across layers  
- Propagates attention from input tokens to output tokens  
- Computes a global attention map

### ✅ 2. **Attention Flow**  
- Proposed in the paper:
  **"Quantifying Attention Flow in Transformers"**   
- Treats attention like flow in a graph  
- Computes how much each input token contributes to a specific output token

📄 **Paper Reference:**  
🔗 https://arxiv.org/abs/2005.00928

---

## 🎥 Presentation

You can find a summary of this project, visual results, and explanations in the final presentation:

👉 **[Access the presentation here](./pres_finale.pptx)**  
