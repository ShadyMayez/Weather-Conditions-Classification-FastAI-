# Weather Conditions Classification using FastAI

## Project Overview and Purpose
This project implements an automated system to classify different weather conditions from images. By leveraging the FastAI framework and pre-trained deep learning models, the system can identify a wide range of meteorological phenomena such as dew, hail, frost, rain, and sandstorms. Such technology is essential for autonomous vehicles, smart city infrastructure, and automated weather monitoring stations.

## Key Technologies and Libraries
- **Deep Learning Framework**: `fastai` (v2)
- **Base Architecture**: `FastAI.vision` models
- **Data Manipulation**: `pandas`, `numpy`
- **Metrics**: Accuracy, error_rate

## Methodology and Workflow
### 1. Data Pipeline & Preprocessing
- **Dataset**: A multi-class weather dataset containing approximately 6,862 images.
- **ETL Process**: Systematically gathered file paths and labels from nested directories into a structured Pandas DataFrame.
- **Loading**: Utilized `ImageDataLoaders.from_df` with a 20% validation split and standardized image resizing to 224x224 pixels.

### 2. Model Development
- **Framework**: Built using the FastAI `vision_learner` (or `cnn_learner`).
- **Optimization**: Implemented FastAI’s automated learning rate finder and "one-cycle" training policy to achieve rapid convergence.


## Results and Insights
- **Performance**: The model demonstrates high accuracy across several distinct weather classes.
- **Error Analysis**: Using `ClassificationInterpretation`, the project identifies "Top Losses"—images where the model was most confident but incorrect—to help refine the dataset.
- **Confusion Matrix**: Highlighting specific class overlaps (e.g., distinguishing between 'rime' and 'frost') to understand visual similarities in weather patterns.

## How to Run
1. **Dataset**: Ensure the weather dataset is located in `/kaggle/input/weather-dataset/dataset`.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
