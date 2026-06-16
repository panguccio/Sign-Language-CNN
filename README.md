# Vision-Based Gesture-to-Speech Translation for Sign Language
---
> Project proposal by: Anna Guccione [IN2300017], Michelangelo Pozza [IN2300012]


Over 5% of the world’s population suffers from a disabling hearing loss. As society rapidly becomes more interconnected, people with disabilities are usually left behind when dealing with new technologies: reducing this gap is therefore essential. Inclusive tools are a necessity for solving these challenges and deserve greater attention.

One active research branch in this area involves the implementation of computer-vision applications for gesture-to-speech translation. Training a Neural Network (NN) on automatic translation of sign language sentences (videos) is quite complex and requires a lot of computational power. Given our limited resources , we decided to implement our custom lightweight NN on a reduced dataset containing only alphabet spelling and numerical gestures. 

The application will involve the training of a Convolutional Neural Network (CNN) on pre-existing dataset for Italian or American Sign Language . Some letters are not included because they require a dynamic hand gesture that cannot be expressed in a static image. Previous scientific studies  that use similar approaches are going to be consulted as a baseline for the realization of our project. 

Ideally, an application will be created that uses the NN found to automatically translate signs from a real-time video. To study the performance of our model, we aim at testing different hyperparameters and compare our results with some well-known architectures such as ResNET50, MobileNetV2, etc.
