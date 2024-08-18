# Comparative Analysis of CNNs for Classifying Alzheimer's Disease 
> Created for the [IJAS](https://sites.google.com/ijas.org/ijas/about) competition in collaboration with Shrihan Tummala @VenkataST

## Acknowledgements
We would like to extend our heartfelt gratitude to our families for their support in helping us pursue our research in this field. And most importantly, we would like to thank our sponsor Dr. Brontman for investing her time into providing valuable and prompt feedback to help us improve our project. This project would not have been possible without her guidance and assistance.

## Purpose
The purpose of this experiment is to determine the ability of various Convolutional Neural Networks (CNNs) in detecting the various stages of Alzheimer’s disease. This research is relevant since it can reveal the merits and shortcomings of various CNNs in this subset of analyzing Alzheimer’s Disease. Alzheimer’s disease is the most common type of dementia and is generally found in elderly citizens. As the world’s population grows, more and more people start becoming vulnerable to this disease. It starts off with memory loss but can result in the patient losing control of their motor skills. It also results in abnormal protein aggregation in the brain, which is present in Magnetic Resonance Imaging (MRIs), allowing CNNs to classify them. This experiment will highlight the best CNN, thereby progressing future development on Alzheimer's and potentially slowing down its spread through early detection and subsequent preventative measures.

## Hypothesis
If the dataset of MRI Segmentation images is fit into a VGG16, ResNet50, DenseNet169, and a Xception convolutional neural network model, then the VGG16 will have highest percent accuracy in detecting the stage of Alzheimer’s. This is because the VGG16 architecture's depth, encompassing numerous layers, contributes to its efficacy in computer vision tasks. The multitude of layers allows for hierarchical feature extraction, where the lower layers capture basic image features, and higher layers detect more complex patterns. This depth of analysis increases the network's capability to model intricate relationships within data while reducing overfitting through regularization techniques. 

## Essential Components
Independent Variable: The various types of CNNs being compared (VGG16, ResNet50, Xception, DenseNet169) in their ability to classify Alzheimer’s disease into Non-Dementia, Very Mild Dementia, Mild Dementia, and Moderate Dementia.
Dependent Variable: The percent accuracy of the CNN models’ predictions measured after the model is trained and run on test data.
Constants: The programming language used (Python 3.10); the computer used; the software the program is run on (Jupyter Notebook); the libraries used (Tensorflow, NumPy, Matplotlib); the 70:20:10 ratio for the training, testing, and validation datasets respectively; the dataset (Balanced Alzheimer MRI Dataset); and the CBAM attention layer for every model.
Comparison Group: A Dense-Net 169 classification algorithm that classifies Alzheimer’s disease into Non-Dementia, Very Mild Dementia, Mild Dementia, and Moderate Dementia. 

## Materials
- 32-bit/64-bit PC (running macOS Sonoma)
- Access to the internet
- Alzheimer MRI Preprocessed Dataset (Retrieved September 9, 2023).
- A web browser such as Google Chrome (Version 120.0.6099.199) that is compatible with Kaggle
- Visual Studio Code Version 1.85
- ResNet50 Model from TensorFlow Version 2.14.0
- VGG16 Model from TensorFlow Version 2.14.0
- Xception Model from TensorFlow Version 2.14.0
- DenseNet169 Model from TensorFlow Version 2.14.0
- A Kaggle Account with a verified phone number

## Procedure
### Balancing Data
1. Download Visual Studio Code from https://code.visualstudio.com/download
2. Press Shift + Command + X to open the extension sidebar
3. Search python
4. Click on the first extension “Python”
5. Click on Install to download the python language
6. Go to https://drive.google.com/file/d/1XDxp6Bw320rT2epj6xX_NTrcRyILmFya/view?usp=sharing
7. Download the code to preprocess the dataset by clicking the download button
8. Open Finder
9. Go to your downloads
10. Right click on the preprocess.py file
11. Hover over the “open with” option
12. Choose Visual Studio Code
13. If Visual Studio Code asks permission to access files in your downloads, click allow
14. Go to https://www.kaggle.com/datasets/ninadaithal/imagesoasis
15. Click the download button in the top right side of screen
16. In Finder, open Downloads
17. Double click the “.zip” file named “archive.zip”
18. Press Option + Command + P to show the path bar on the bottom of the finder window
19. With the Data folder selected (highlighted in blue), right click the “Data” button on the
path bar
20. Click “Copy Data as Pathname”
21. Go back to Visual Studio Code
22. Press Command + Shift + P
23. Type “terminal”
24. Select the “Python: Create Terminal” option
25. In the terminal that appears on the bottom, type “pip install opencv-python”
26. After the command executes, type “pip install albumentations”
27. Click the run button   at the top right of the VS Code Window
28. When prompted in the terminal to enter the path for the dataset, press Command + V
29. Press Enter to make the dataset balanced by augmenting images for the Moderate
Dementia class (488 Images → 4880 Images) and removing excess images for the Non Demented (67,222 Images → 5000 Images), Mild Dementia (5002 Images → 5000 Images), and Very Mild Dementia (13,725 Images → 5000 Images) classes.
30. In Finder, open Downloads
31. Right click the “Data” folder that has just been preprocessed
32. Click ‘Compress “Data”’ to make a zipped version of the balanced dataset
### Kaggle Setup:
33. Register an account on Kaggle:
https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F
34. Click on your profile picture in the top right corner
35. Click on “Settings”
36. Using the “Phone Verify” button, verify your phone number to be able to use an
accelerator
37. Click the following link to open the notebook:
https://www.kaggle.com/code/debarhyaroy11707/roy-tummala-23-24-alzheimer-s-classifi
cation?scriptVersionId=158232378
38. Click the “Copy and Edit” button in the top right corner of the screen
39. Scroll down in the “Notebook” tab on the right side until you find a “Notebook Options” dropdown
40. Open the “Notebook Options” dropdown
41. Choose the GPU T4 x2 accelerator in the “Accelerator” dropdown
### Loading the Dataset:
42. In the “Notebook” tab, click the upload data button next to the + Add Data button
43. Click Browse Files
44. Go to your Downloads folder
45. Select the zipped Data.zip file
46. Click Open
47. Fill in a title for the dataset
48. Click Create
49. Once the data has been uploaded, click on the dataset in the sidebar to show the “Data”
  folder
50. Hover over the the Data folder to see a copy button
51. Click the copy button to copy the path for that directory
52. Go to cell 2 in Kaggle
53. On the second line, select the text inside the quotes
54. Press Command + V
### Running the code:
55. Click on cell 1
to the right of it
56. Click the Run button   to import the matplotlib library for creating graphs, and several modules in the TensorFlow library including the CNN models (ResNet50, VGG16, Xception, DenseNet169)
57. Click on cell 2
58. Click the Run button to load the dataset and split it into 70% for training, 20% for testing,
and 10% for validation
59. Click on cell 3
60. Click the Run button to create several functions for the CBAM attention mechanism
61. Click on cell 4
62. Click the Run button to merge all the layers to be added after the pre trained model
63. Click on cell 5
64. Click the Run button to create a function that can graph the training and validation
accuracies and losses
65. Click on cell 6
66. Click the Run button to create a function that will test the model and output a test
accuracy
67. Click on cell 7
68. Click the Run button to create the early stopping mechanism
69. Click on cell 8
70. Click the Run button to download the weights for the ResNet50 model and train it
71. Click on cell 9
72. Click the Run button to test the ResNet50 model on the testing dataset
73. Record the test accuracy for the ResNet50 model
74. Click on cell 10
75. Click the Run button to produce the graphs for the training and validation accuracies and
losses for the ResNet50 model
76. Repeat steps 68-74 for cells 11-13, 14-16, and 17-19 to train and evaluate the
DenseNet169, VGG16, and Xception models respectively
### Conclusion
The purpose of this experiment was to identify which CNN model would perform the best in classifying the different stages of Alzheimer’s based on an MRI. As the life expectancy of humans increases, there has also been a corresponding increase in seniors aged 65+. The Alzheimer’s Disease International Organization predicts that the number of people worldwide suffering from dementia is expected to approximately double every 20 years, going from 55 million in 2020 to 139 million in 2050 (Alzheimer’s Disease International - dementia statistics, n.d.). As such, due to the detrimental impact Alzheimer’s poses to the geriatric population, identifying the disease accurately has become a more pressing priority. The early detection of Alzheimer’s is also crucial in prompting intervention that can slow the disease from progressing.

The hypothesis was that the VGG16 would perform with the highest accuracy out of the four CNNs due to its complex layers and extensive feature extraction which allows the network to discern complex patterns while mitigating overfitting through regularization techniques. Although this prediction is supported by Figure 16, as VGG16 achieved the highest accuracy of 94.83%, it is important to acknowledge that every single algorithm achieved accuracy above 90%, reinforcing their reliability in identifying Alzheimer’s. Additionally, in Figure 12, the training and validation loss curves for VGG16 are closer together than those for ResNet50 in Figure 8. This suggests that VGG16 may be less prone to overfitting, as a large gap between the training and validation loss curves is generally an indicator of overfitting.

An experimental error that could potentially impact the data involves the approach of replicating the Moderate Dementia class several times with minor modifications to augment the data. While the technique helped artificially increase the size of the class in order to balance it with the other classes, it raises the possibility that the model could struggle to recognize complexities in more diverse datasets in the real world due to the limited variety of images it was trained on. In future experiments, this source of error could be eliminated by more thorough data augmentation techniques or through another source of MRI images for Moderate Dementia that could be added onto the original dataset.

DenseNet169’s success was quite unexpected as it is a fairly simple and straightforward model because its architecture only combines features from all preceding layers using dense blocks to find patterns. This type of architecture results in it having comparatively fewer parameters, with 14.3 million parameters total. When compared with Xception, it has 8.6 million less parameters, which signifies less weights that can be learnt during training. Thus, it was surprising to see that it outperformed Xception by more than 2%. Moreover, although VGG16 (138.4 million parameters) had more than 5 times the parameters than ResNet50 (25.6 million parameters) they both performed nearly the same. The comparisons above indicate parameters are not proportional to accuracy which is contrary to logic as generally, more parameters means more weights where it can learn patterns and store them in the convolutional layers.

This experiment’s real-world application involves the potential for enhancement on Alzheimer’s diagnosis accuracy. There have already been various studies, among which include one from Massachusetts General Hospital, where researchers attempted to develop an accurate method of detecting Alzheimer’s through deep learning. The increased relevancy of AI in the real world has led to an explosive growth in AI based studies in almost every single field that is compatible with it. While each individual study may contribute to notable improvements, the cumulative effect of refining the diagnostic approaches taken to identify Alzheimer’s through AI has the potential to save countless lives. Although the 1% or so differences in accuracies between the models may seem small, when scaled to the over 55 million people suffering from Alzheimer’s in the world, it could result in over half a million misclassifications.

Further research can be conducted to this experiment by integrating all of our 4 CNNs into what is known as an ensemble model. This technique has the potential of improving the accuracy further by combining the four CNNs into one model in the prediction stage for the testing datasets. Ensemble models can yield improved results as this approach capitalizes on the strengths of individual algorithms to enhance the collective performance. For instance, combining parts of a CNN adept at identifying early-stage Alzheimer’s symptoms with another that is more proficient at detecting more moderate and later stages could lead to the algorithms complementing each other’s weaknesses to create a more accurate diagnosis. In addition, a larger and more diverse dataset that includes statistics like the age of the patient would improve the accuracy of the model in the real world. Finally, an attempt could be made to visualize the areas of the MRI that the CNN focuses on through the implementation of attention masks. This visualization could help doctors learn new patterns for identifying Alzheimer’s and also aid in directing the CNN to focus on specific regions of the brain associated with Alzheimer’s.
