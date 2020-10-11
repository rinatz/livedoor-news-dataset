# livedoor-news-dataset

[Livedoor ニュースコーパス] を使ったニュースの分類器です。

## 必要なもの

- Python >= 3.6
- Poetry
- MeCab
- NEologd

## 準備

```shell
$ make init
```

## データセットのダウンロード

データセットをダウンロードし、形態素解析を行います。

```shell
$ make download
```

## モデルの作成

コーパスを使ってディープラーニングでモデルを作成します。

```shell
$ make train
```

## 分類器を試す

Web ページを開きます。

```shell
$ make serve
```

http://localhost:8501 にアクセスすると Web ページが表示されますので、適当な文章を入力して下さい。下部に判定結果がグラフで表示されます。

[Livedoor ニュースコーパス]: http://www.rondhuit.com/download.html#ldcc
