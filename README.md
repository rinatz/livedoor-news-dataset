# news_classification

[Livedoor ニュースコーパス] を使ってニュース記事の分類器を作成します。

## 必要なもの

- Python 3.7
- MeCab
- NEologd (MeCab 辞書)

## インストール

```shell
$ pipenv sync
```

## モデルの作成

コーパスを使ってディープラーニングでモデルを作成します。

```shell
$ pipenv run python -m news_classification create-model
```

## 分類器を試す

サーバを起動します。

```shell
$ pipenv run python -m news_classification server
```

http://localhost:8000 にアクセスすると UI ページが表示されますので、適当な文章を入力して下さい。下部に判定結果がグラフで表示されます。

[Livedoor ニュースコーパス]: http://www.rondhuit.com/download.html#ldcc