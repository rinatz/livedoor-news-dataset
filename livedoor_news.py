import keras
import pickle
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# livedoorニュースコーパスのダウンロードと解凍
url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
extracted_dir = keras.utils.get_file("ldcc-20140209", origin=url, extract=True)

data_dir = f"{extracted_dir}/text"  # 解凍されたテキストフォルダ
batch_size = 32
validation_split = 0.2

# データセットの作成
train_ds = keras.utils.text_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    validation_split=validation_split,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    validation_split=validation_split,
    subset="validation",
    seed=42,
)

# ラベル情報の取得
class_names = train_ds.class_names

# トークナイザのセットアップ
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# トークナイザの学習
texts = [text.numpy().decode("utf-8") for text, label in train_ds.unbatch()]
trainer = WordLevelTrainer(vocab_size=20000, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train_from_iterator(texts, trainer)

# パディング処理を含むデータ前処理関数
max_sequence_length = 100

def preprocess_dataset(dataset, tokenizer, max_len):
    texts = [text.numpy().decode("utf-8") for text, label in dataset.unbatch()]
    labels = [label.numpy() for text, label in dataset.unbatch()]
    tokenized = [tokenizer.encode(text).ids for text in texts]
    padded = keras.utils.pad_sequences(tokenized, maxlen=max_len, padding='post', truncating='post')
    return np.array(padded), np.array(labels)

# データの前処理
X_train, y_train = preprocess_dataset(train_ds, tokenizer, max_sequence_length)
X_val, y_val = preprocess_dataset(val_ds, tokenizer, max_sequence_length)

# データの形状確認
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# モデル構築
num_classes = len(class_names)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=20000, output_dim=128),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax'),
])

# モデルコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_val, y_val))

# モデル、トークナイザ、ラベル情報の保存
model.save("news_classification_model.h5")
tokenizer.save("news_tokenizer.json")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("モデル、トークナイザ、ラベルエンコーダの保存が完了しました。")
