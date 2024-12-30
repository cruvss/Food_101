# Food_101
This repository contains an implementation of a Food-101 image classifier using the DeiT (Data-efficient Image Transformer) model . Currently, the model achieves a test accuracy of 78%. Further fine-tuning of the model is planned to improve the performance.

# Dataset: Food-101
The Food-101 dataset is a large-scale collection of food images. It contains 101 classes, each representing a different type of dish, with 1,000 images per class. The dataset is divided into:

  - **Training set:**   70,700 images (700 images per class)
  - **Test set:**       30,300 images (300 images per class)

# Current Progress
## v1.0
- Model Used: DeiT (Data-efficient Image Transformer)
- Accuracy Achieved: 78% on the test set
- Status: Fine-tuning of the final model is yet to be completed to further optimize accuracy.

## v1.1
- Model Used: swin-transformer
- Accuracy Achieved: 85% on the test set

# Important Links

- [Food-101 Dataset](https://www.kaggle.com/datasets/dansbecker/food-101)
- [DeiT Documentation](https://huggingface.co/docs/transformers/en/model_doc/deit)
