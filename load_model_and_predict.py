import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument("seed_text");ap.add_argument("--words",type=int,default=30);ap.add_argument("--artifacts",type=Path,default=Path("artifacts"));ap.add_argument("--temperature",type=float,default=0);ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if not 1<=args.words<=1000 or args.temperature<0:ap.error("Words must be 1..1000 and temperature nonnegative")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer=tokenizer_from_json((args.artifacts/"tokenizer.json").read_text());context=json.loads((args.artifacts/"metadata.json").read_text())["context"]
    model=tf.keras.models.load_model(args.artifacts/"model.keras");rng=np.random.default_rng(args.seed);text=args.seed_text
    for _ in range(args.words):
        tokens=tokenizer.texts_to_sequences([text])[0]
        if not tokens:raise ValueError("Seed text contains no tokens")
        scores=np.asarray(model.predict(pad_sequences([tokens],maxlen=context,padding="pre"),verbose=0)[0],dtype=float);scores[0]=0
        if args.temperature:
            scores=np.exp(np.log(np.maximum(scores,1e-30))/args.temperature-np.max(np.log(np.maximum(scores,1e-30))/args.temperature));scores[0]=0;scores/=scores.sum();index=int(rng.choice(len(scores),p=scores))
        else:index=int(scores.argmax())
        text+=' '+tokenizer.index_word.get(index,'<OOV>')
    print(text)
if __name__=="__main__":main()
