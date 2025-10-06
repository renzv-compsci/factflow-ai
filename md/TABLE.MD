# Feature Engineering Table for FactFlow AI

| Column Index | Feature          | How to Use                      |
|--------------|------------------|---------------------------------|
| 2            | Statement        | Main text, always include       |
| 3            | Subject          | Combine with text, or encode    |
| 4            | Speaker          | Combine or encode               |
| 5            | Role             | Combine or encode               |
| 6            | Location         | Combine or encode               |
| 7            | Party            | Combine or encode               |
| 8–12         | Numbers          | Use directly as numeric features|
| 13           | Context/Source   | Combine with text               |

---

**Usage Tips:**
- Do not include the ID column (0) or original label column (1) as features.
- For text features, you can concatenate multiple columns to create a richer input for your model.
- For categorical features (subject, speaker, etc.), consider encoding them (one-hot or label encoding).
- For numeric features (columns 8–12), include them directly in your feature set.
- Always check for missing values before combining or encoding columns.