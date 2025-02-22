# **Breast Ultrasound U-Net Model**

## **Overview**

This project applies deep learning-based segmentation to identify tumor regions in breast ultrasound images. Using the U-Net architecture, a widely used convolutional neural network (CNN) for medical image segmentation, the model learns to distinguish between tumor and non-tumor areas, generating binary segmentation masks. 

By leveraging pixel-wise classification, this approach provides a more detailed analysis of benign and malignant tumors, improving early detection capabilities. The results of this study highlight the potential for AI-driven diagnostic tools to assist radiologists in detecting breast cancer more efficiently.

---

## **Dataset and Features**

The dataset used for this project is the Breast Ultrasound Images Datase (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download), sourced from Kaggle. It consists of ultrasound scans labeled into three categories: benign tumors, malignant tumors, and normal breast tissue. Only benign and malignant images were used for segmentation.

- **Total Images:** 1312
- **Classes:**
  - **Benign Tumors:** 437 images, with 454 corresponding segmentation masks.
  - **Malignant Tumors:** 210 images, with 211 corresponding segmentation masks.
- **Image Format:** `.png`
- **Masks:** Each tumor image has an associated mask (`_mask.png`), where white pixels indicate the tumor region.

---

## **Deep Learning Pipeline**

### 1. **Data Preprocessing**
   - Loaded ultrasound images and corresponding masks.
   - Normalized pixel values to the range [0,1] to stabilize training.
   - Resized images to 224×224 pixels for consistency.

### 2. **Model Architecture: U-Net**
   - Implemented a U-Net model, designed for biomedical image segmentation.
   - **Encoder:** Extracts feature maps using convolutional layers and max pooling.
   - **Bottleneck:** Captures high-level representations of tumor structures.
   - **Decoder:** Upsamples feature maps and concatenates them with earlier layers for fine-grained segmentation.

### 3. **Training and Optimization**
   - **Loss Function:** Binary Cross-Entropy.
   - **Optimizer:** Adam.
   - **Metrics:** Accuracy and Dice Coefficient.
   - **Training Strategy:**
     - Data split: **80% training, 20% validation**.
     - Batch size: **16**.
     - Early Stopping to prevent overfitting.

---

## **Results and Performance**

| Metric                  | Value  |
|-------------------------|--------|
| **Validation Accuracy** | 94.16% |
| **Validation Loss**     | 0.1592 |
| **Mean Dice Score**     | 0.3986 |

The model achieved high validation accuracy (94.16%), indicating strong generalization. However, the Mean Dice Coefficient (0.3986) suggests that the predicted masks do not perfectly overlap with ground truth tumor masks, indicating room for improvement in fine-tuning the segmentation boundaries.

---

## **Next Steps and Areas for Improvement**

1. **Loss Function Enhancement:** Implement Dice Loss or IoU Loss to optimize segmentation boundaries.
2. **Data Augmentation:** Apply rotation, flipping, and contrast adjustments to increase dataset variability.
3. **Alternative Architectures:** Experiment with Attention U-Net or DeepLabV3+ for improved segmentation.
4. **Post-Processing Techniques:** Use morphological operations to refine mask predictions.
5. **Integration with Classification:** Apply segmented tumor regions to a CNN classifier for benign vs. malignant diagnosis.

---

## **Author**

**Justin Ho**  
- **LinkedIn**: [linkedin.com/in/justin-ho-4a6157285](https://www.linkedin.com/in/justin-ho-4a6157285/)  
- **GitHub**: [github.com/justinh128](https://github.com/justinh128)
- **Email**: justinh128@gmail.com

---

## **Acknowledgments**
- **Dataset**: [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download) by Arya Shah.  
- Special thanks to open-source contributors and medical imaging researchers whose work inspires advancements in AI-driven cancer diagnostics.
