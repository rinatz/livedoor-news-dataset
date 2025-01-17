import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
import keras
import pickle

# フォルダからlivedoorニュースコーパスを読み込む関数
def load_livedoor_news_corpus(data_dir):
    data = []
    labels = []

    # サブフォルダ（カテゴリ名）を取得
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):  # サブフォルダのみ処理
            for file_name in os.listdir(category_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(category_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if len(lines) > 2:  # 最低限のフォーマット確認
                            text = "".join(lines[2:])  # 本文部分を結合
                            data.append(text)
                            labels.append(category)
    return data, labels

# データの読み込み
data_dir = "livedoor_news"  # livedoorニュースコーパス解凍後のフォルダ
texts, labels = load_livedoor_news_corpus(data_dir)

# pandas DataFrameに変換
df = pd.DataFrame({"text": texts, "label": labels})

# ラベルエンコード
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Hugging Face Tokenizerのセットアップ
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# トークナイザの学習
trainer = WordLevelTrainer(vocab_size=20000, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train_from_iterator(df["text"], trainer)

# トークン化とパディング処理
max_sequence_length = 100

def tokenize_and_pad(texts, tokenizer, max_len):
    tokenized = [tokenizer.encode(text).ids for text in texts]
    padded = keras.utils.pad_sequences(tokenized, maxlen=max_len, padding='post', truncating='post')
    return padded

X_padded = tokenize_and_pad(df["text"], tokenizer, max_sequence_length)
y_encoded = df["label_encoded"].values

# モデル構築
num_classes = len(label_encoder.classes_)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=20000, output_dim=128, input_length=max_sequence_length),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax'),
])

# モデルコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(X_padded, y_encoded, epochs=10, batch_size=32, validation_split=0.2)

# モデルの保存
model.save("news_classification_model.keras")

# トークナイザの保存
tokenizer.save("news_tokenizer.json")

# ラベルエンコーダの保存
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("モデル、トークナイザ、ラベルエンコーダの保存が完了しました。")
