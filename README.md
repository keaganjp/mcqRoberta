
# Multiple Choice Question Answering with RoBERTa

This project fine-tunes the `RoBERTa` model on the AI2 Reasoning Challenge (ARC) dataset for multiple choice question answering. It leverages pre-trained models on Hugging Face in an attempt to solve the ARC challenge. 

## Table of Contents

- Project Overview
- Installation
- Usage
- Project Structure
- Contributing
- License

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mcq-roberta.git
   cd mcq-roberta
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Fine-tuning on ARC Dataset

1. Run the fine-tuning script:
   ```bash
   python finetune.py --dataset ai2_arc,ARC-Challenge --model roberta-base --batch_size 8 --num_epochs 3 --learning_rate 2e-5

   ```

### Evaluating the Model


## Project Structure

```
mcq-roberta/
│
├── finetune.py             # Script for fine-tuning on the ARC dataset
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Customization

### Adjust Hyperparameters

You can customize the training process by adjusting hyperparameters such as learning rate, batch size, and number of epochs in the `finetune.py` script.

### Use Different Models

To use a different model, modify the `--model` argument in the command line. For example, to use `roberta-large`:

```bash
python finetune.py --dataset ai2_arc,ARC-Challenge --model roberta-large --batch_size 4 --num_epochs 3 --learning_rate 1e-5
```

## Learnings and Future Work

This project was an attempt to utilize a pre-trained Roberta model to solve the ARC challenge. In the future I will be using the ARC challenge custom corpus to pretrain the model and continue fine tuning. This project was executed with limited compute- larger models could be leveraged to train these models.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---