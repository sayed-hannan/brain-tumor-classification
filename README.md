# Brain Tumor Classification

This project aims to classify brain tumor images using the VGG-16 transfer learning model. With the utilization of the VGG-16 architecture, the project has achieved near to 90% accuracy in classifying brain tumor images into different categories.

## Installation

To get started, you can clone this repository using the following command:

```
git clone https://github.com/sayed-hannan/brain-tumor-classification
```

After cloning the repository, create a virtual environment using Python's `venv` module:

```
python -m venv venv
```

Activate the virtual environment:

For Windows:

```
env\Scripts\activate
```

For Unix/macOS:


```
venv\Scripts\activate
```

## Usage

Once the virtual environment is activated, navigate to the `src/` directory:

```
cd src/
```

Here, you will find the relevant scripts for model training and evaluation.

### Model Training

For model training, run the appropriate script. Since relative paths are used, ensure you are in the `src/` directory:

```
python trainer.py
```

### Colab Notebook

To access a Colab notebook for training and fine-tuning the VGG model, simply upload it to Google Colab and utilize it. The notebook includes VGG model training, fine-tuning, and oversampling for minor classes.

### Tensorboard Logs

Tensorboard logs for VGG-16 training and fine-tuning are available in the `tensorboard_logs/` directory.

## Requirements

Ensure you have the following dependencies installed:

- matplotlib
- tensorflow

You can install them using pip:


pip install matplotlib tensorflow

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests or open issues for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, feedback, or inquiries, please contact [Sayed Hannan](https://github.com/sayed-hannan).

## Acknowledgments

- Special thanks to the developers and contributors of the VGG-16 architecture and TensorFlow framework.

