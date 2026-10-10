# Word prediction experiment

Train first with `python word_predictor.py --help`, then use `python load_model_and_predict.py --help`. No pretrained model is shipped. Training writes model.keras, the exact tokenizer and preprocessing metadata together. Validation is sentence-separated; seeds and bounded sampling make the experiment repeatable. Keep those artifacts together when running inference.
