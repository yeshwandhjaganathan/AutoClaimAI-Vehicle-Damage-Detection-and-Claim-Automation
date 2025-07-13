AutoClaimAI: End-to-End Vehicle Damage Detection and Insurance Claim Automation

![AutoClaimAI Banner](https://dummyimage.com/1200x300/000/fff&text=AutoClaimAI+-+Vehicle+Damage+Detection+and+Claim+Automation)

---

## 🚗 Overview

**AutoClaimAI** is an AI-powered framework to automate vehicle insurance claim assessment.  
It integrates **YOLOv5** for damage detection, **DenseNet121** for severity classification, **ImageHash** for fraud detection, and **SHAP** for explainability.

This project is intended for **educational and demonstration purposes** only.

---

## ✨ Key Features

✅ Detects and localizes vehicle damages in images  
✅ Classifies damage severity (*Minor*, *Moderate*, *Severe*)  
✅ Checks for duplicate or fraudulent claims  
✅ Generates explainable SHAP visualizations  
✅ Exports results to CSV reports

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow** (DenseNet121)
- **PyTorch & Ultralytics YOLOv5**
- **ImageHash**
- **SHAP**
- **OpenCV**
- **Google Colab**

---

## 📂 Project Structure

AutoClaimAI/ ├── AutoClaimAI_Notebook.ipynb   # Full Google Colab notebook with all code cells ├── README.md                    # This file ├── LICENSE                      # License file (CC BY-NC 4.0) └── sample_outputs/              # Example result CSV and images (optional)

---

## 🚀 Getting Started

1. **Open Google Colab**
   - Upload the `AutoClaimAI_Notebook.ipynb`
2. **Install dependencies**
   - The notebook will automatically install required packages.
3. **Upload your vehicle images**
   - Use the upload cell.
4. **Run each section:**
   - Damage Detection
   - Severity Classification
   - Fraud Check
   - SHAP Explainability
5. **Export CSV report**
   - Results will be saved to a downloadable CSV file.

---

## 📈 Example Output

Image: car_damage_1.jpg

Damage Detection: Objects Detected: car, person Confidence: [0.85, 0.67] Bounding Boxes: [[x1,y1,x2,y2], ...]

Severity Classification: Predicted Severity: Moderate Confidence: 0.74

Fraud Check: Duplicate Status: Unique

Explainability: SHAP plot generated.


---

## 🔒 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

[Read the License](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [TensorFlow](https://www.tensorflow.org/)
- [SHAP](https://github.com/slundberg/shap)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)

---

## 📬 Contact

For any queries, feel free to raise an issue or discussion in this repository.

---

> 🚀 **Let's automate vehicle claim assessments with AI!**

