import logging

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)

if __name__ == '__main__':
    # trains a bert masked language model using sentences that concatenate the
    # filename with its own caption for the training set.
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)

    model_args = LanguageModelingArgs()

    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.save_steps = 100000
    model_args.num_train_epochs = 2
    model_args.save_model_every_epoch = True
    model_args.dataset_type = "simple"
    model_args.vocab_size = 60000
    model_args.silent = False
    model_args.output_dir = 'bert_lm'

    train_file = "data/train_bert_lm.txt"

    model = LanguageModelingModel(
        "bert", None, args=model_args, train_files=train_file
    )

    model.train_model(train_file)
