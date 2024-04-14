# PredNet Implementation in TensorFlow
This repository hosts a modern TensorFlow implementation of the PredNet architecture, originally proposed for next-frame video prediction. Our implementation enhances the modular and reusable aspects of PredNet using TensorFlow's latest features, making it suitable for both research and application development. Additionally, we provide a direct comparison with the original PredNet model and include a custom data pipeline for handling video data, particularly from the KITTI dataset.
## Repository Structure
* '**PredNet.py**' - Contains the modular TensorFlow/Keras implementation of the PredNet model.
* '**model.py**' - Hosts the '**PredNetModel**', a high-level model class for integrating the PredNet cells into a trainable model
* '**datapipeline_tf_kitti'** - A directory containing the TensorFlow data pipeline setup for processing raw picture data into a format suitable for PredNet training.
* '**original_prednet.py**' - The original PredNet implementation as provided in the [PredNet paper](https://coxlab.github.io/prednet/).
* '**PredNet_Comparison.ipynb**' - A Jupyter notebook hosted on Google Colab that demonstrates the use of the new modular PredNet, comparing its performance and functionality against the original implementation.

## Features
* **Modular Implementation**: Redefines the PredNet architecture within the TensorFlow 2 ecosystem, emphasizing reusability and easy integration with modern deep learning workflows.
* **Custom Data Pipeline**: Includes a TensorFlow-based pipeline for processing video data, making it easy to prepare datasets like KITTI for training. Ensure the data is formatted correctly (e.g., folders of images representing video sequences).
* **Performance Comparison**: Provides empirical comparisons between our implementation and the original model, ensuring that enhancements or modifications do not compromise the core functionalities.

## Usage

### Running in Google Colab
Since the code is primarily written and tested in Google Colab, open the provided PredNet_Comparison.ipynb notebook and follow the steps there. This method ensures that all dependencies and the environment are correctly configured for immediate use.
### Running Locally
If you prefer to run the code locally:

1. **Clone the repository**:

```
git clone https://github.com/karioth/PredNet_tf.git
cd PredNet_tf

```
2. **Install Dependencies**:

    * Ensure you have Python and TensorFlow installed, and install any additional required packages.
    
### Data Preparation
Use the scripts in '**datapipeline_tf_kitti**' (see '**testing_kitti_tf_data.ipynb**' as an example) to process your video data. These scripts will format the data correctly for training with the PredNet model.

#### Dataset Requirements

##### Image Size Constraints

When training the PredNet model on a new dataset, it is crucial to ensure that the dimensions of your input images are compatible with the model's architecture. Specifically, each dimension (width and height) of the images must be divisible by `2^(number of layers - 1)`. This constraint arises due to the cyclical 2x2 max-pooling and upsampling operations used within the network.

    
### Modifying the Model
To adjust the model architecture or training process, modify the '**PredNetModel**' class in '**model.py**'. For custom data inputs or formats, update the data pipeline scripts in the '**datapipeline_tf_kitti**' directory.
