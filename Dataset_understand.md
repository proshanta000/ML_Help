# 🧠 ML Model Selection Cheat Sheet (Bangla + English)

## 📌 Step 1: Identify Problem Type
| Problem Type        | বাংলা অর্থ                     | Example                                | Suggested Models |
|---------------------|--------------------------------|----------------------------------------|------------------|
| **Regression**      | সংখ্যাগত মান ভবিষ্যদ্বাণী       | House price, Salary prediction         | Linear Regression, Decision Tree Regressor, Random Forest Regressor |
| **Classification**  | ক্যাটাগরি/লেবেল ভবিষ্যদ্বাণী    | Spam vs Not Spam, Disease detection    | Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network |
| **Clustering**      | ডেটাকে গ্রুপে ভাগ করা          | Customer Segmentation, Document grouping | KMeans, Hierarchical, DBSCAN |
| **Dimensionality Reduction** | ভ্যারিয়েবল কমানো       | Visualization, Noise reduction         | PCA, t-SNE, LDA |
| **Ensemble**        | একাধিক মডেল একত্রে ব্যবহার     | Competition-winning models             | Random Forest, Gradient Boosting (XGBoost, LightGBM), Stacking |

---

## 📌 Step 2: Dataset Size & Complexity
| Dataset Size / Nature | বাংলা অর্থ       | Suggested Models |
|-----------------------|-----------------|------------------|
| **Small (কম ডেটা)**  | অল্প ডেটাসেট     | Logistic Regression, Decision Tree |
| **Medium (মাঝারি)**  | মাঝারি আকার     | Random Forest, SVM |
| **Large (বড় ডেটাসেট)** | বিশাল ডেটা    | XGBoost, LightGBM, Deep Learning (ANN, CNN, RNN) |
| **High Dimensional**  | অনেক ফিচার      | PCA + Classifier/Regressor |

---

## 📌 Step 3: Quick Tips
- ✅ Always start with a **simple model** (baseline তৈরি করো)।  
- ✅ Measure **performance** (Accuracy, RMSE, F1-score ইত্যাদি)।  
- ✅ If simple model ভালো না করে → Complex model use করো।  
- ✅ Unsupervised হলে আগে **Clustering/Dimensionality Reduction** apply করো।  

---

## 📝 Example Workflow
1. **Target আছে?**  
   - হ্যাঁ → **Supervised**  
   - না → **Unsupervised**  
2. যদি supervised হয় → **Regression নাকি Classification** সেটা ঠিক করো।  
3. Dataset size দেখে **Simple → Complex model** use করো।  
4. Performance মেপে Compare করো।  

---

✍️ Created for **easy ML model selection** (বাংলা + English)
