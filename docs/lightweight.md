# How can a model classified as lightweight?

## Introduction

To classify a model as lightweight, we need to consider the following factors:

- **Model size**: The size of the model should be small enough to be deployed on edge devices.
- **Model complexity**: The model should be simple enough to be deployed on edge devices.
- **Model performance**: The model should be accurate enough to be used in real-life applications.
- **Model speed**: The model should be fast enough to be used in real-time applications.

Depend on the application, the value of these factors may vary. For example, if the application is a real-time 
application, the model speed is more important than the model size. If the application is a mobile application, the 
model size is more important than the model speed.

In this project, I set the following criteria to classify a model as lightweight:

- **Model size**: The size of the model should be less than 100 MB.
- **Model complexity**: The model should be a single model, not an ensemble model.
- **Model performance**: The model should have a top-1 accuracy of at least 70%.
- **Model speed**: The model should be able to run at least 10 FPS on a CPU.
