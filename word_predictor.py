"""Sentence-held-out training; artifacts include the exact tokenizer and context."""
import argparse,json,math
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--corpus",type=Path,default=Path("bible.txt"));ap.add_argument("--output",type=Path,default=Path("artifacts"));ap.add_argument("--samples",type=int,default=5000);ap.add_argument("--context",type=int,default=40);ap.add_argument("--epochs",type=int,default=15);ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if min(args.samples,args.context,args.epochs)<1:ap.error("Samples, context and epochs must be positive")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tf.keras.utils.set_random_seed(args.seed);rng=np.random.default_rng(args.seed)
    sentences=[''.join(c for c in s if c.isalpha() or c.isspace()).strip() for s in args.corpus.read_text().split('.')];sentences=[s for s in sentences if len(s.split())>1]
    if len(sentences)<10:raise ValueError("At least ten sentences required")
    order=rng.permutation(len(sentences));cut=int(.85*len(order));train=[sentences[i] for i in order[:cut]];validation=[sentences[i] for i in order[cut:]]
    tokenizer=Tokenizer(oov_token="<OOV>");tokenizer.fit_on_texts(train)
    def sequences(corpus,limit):
        # Reservoir sampling bounds memory even for large corpora.
        sample=[];seen=0
        for tokens in tokenizer.texts_to_sequences(corpus):
            for i in range(1,len(tokens)):
                seq=tokens[max(0,i-args.context):i+1];seen+=1
                if len(sample)<limit:sample.append(seq)
                else:
                    j=int(rng.integers(seen))
                    if j<limit:sample[j]=seq
        padded=pad_sequences(sample,maxlen=args.context+1,padding="pre");return padded[:,:-1],padded[:,-1]
    x,y=sequences(train,args.samples);vx,vy=sequences(validation,max(100,args.samples//5));vocab=len(tokenizer.word_index)+1
    model=tf.keras.Sequential([tf.keras.layers.Input((args.context,)),tf.keras.layers.Embedding(vocab,100),tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),tf.keras.layers.Dense(vocab,activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(x,y,validation_data=(vx,vy),epochs=args.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)])
    loss,accuracy=model.evaluate(vx,vy,verbose=0);args.output.mkdir(parents=True,exist_ok=True);model.save(args.output/"model.keras")
    (args.output/"tokenizer.json").write_text(tokenizer.to_json())
    (args.output/"metadata.json").write_text(json.dumps({"context":args.context,"seed":args.seed,"validation_loss":loss,"perplexity":math.exp(min(loss,700)),"validation_accuracy":accuracy,"samples":len(y)},indent=2))
    print((args.output/"metadata.json").read_text())
if __name__=="__main__":main()
