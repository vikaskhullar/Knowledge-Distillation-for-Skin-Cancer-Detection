Title
Pretrained Deep Trained Model through Knowledge Distillation for Skin Cancer Detection for Low-Resource Devices

Background / Problem
Skin cancer is a highly prevalent global malignancy, especially concerning for fair-skinned populations.
Early and accurate diagnosis is critical for improving survival rates.
Traditional diagnostic methods (e.g., biopsies, medical imaging) are invasive and can be inaccurate or prone to error.

Objective / Proposed Solution
Develop an automatic, efficient skin lesion classification system using a Knowledge Distillation (KD) based Deep Learning (DL) framework.
Use a high-capacity "teacher" model to transfer knowledge to a lightweight "student" model.
Achieve similar accuracy with reduced computational complexity, making it suitable for low-resource devices (edge/clinical environments).

Methods
Implemented various CNN architectures as both teacher and student models: DenseNet, EfficientNet, NASNetMobile, ResNet, VGG16, VGG19, and Xception.
Trained and tested using skin image datasets.

Key Results
Best Teacher Model: DenseNet performed best with 97.3% training and 81.1% validation accuracy.
Best Student Model: EfficientNet showed superior performance after distillation, achieving 97.8% training and 81.4% validation accuracy, with minimal loss.
EfficientNet effectively retained the teacher's knowledge, achieving a macro-averaged F1-score of 0.84.
The framework successfully minimized memory usage and model size while maintaining performance metrics.

Conclusion
The proposed KD framework is effective for creating compact, efficient models.
It is suitable for deployment on resource-constrained devices for real-time skin cancer screening.
