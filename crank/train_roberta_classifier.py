import logging
from multiprocessing import freeze_support
from urllib.parse import unquote

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)


def url_to_filename(url):
    return unquote(url[url.rfind('/') + 1:url.rfind('.')]).replace('_', ' ')


if __name__ == '__main__':
    # trains a sentence pair classifier based on xlm-roberta
    # it classifies pairs of filename and caption
    # as either matching or non-matching

    freeze_support()
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    batch_size = 64
    model_args = ClassificationArgs(num_train_epochs=2,
                                    output_dir='roberta_classifier',
                                    overwrite_output_dir=True,
                                    save_steps=100000,
                                    save_model_every_epoch=True,
                                    train_batch_size=batch_size,
                                    lazy_loading=True,
                                    lazy_delimiter='\t',
                                    lazy_text_a_column=0,
                                    lazy_text_b_column=1,
                                    lazy_labels_column=2,
                                    lazy_loading_start_line=1)

    # model = 'xlm-roberta-large'
    model_name = 'xlm-roberta-base'

    model = ClassificationModel("auto", model_name, args=model_args)

    model.train_model('data/train_roberta_classifier.tsv')
