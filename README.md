# news_classifier

[Livedoor ニュースコーパス] を使ったニュースの分類器のリファレンス実装です。

## 必要なもの

- Python 3.7
- Pipenv
- MeCab
- NEologd (MeCab 辞書)

## 準備

```shell
$ pipenv sync
$ pipenv run init
```

## モデルの作成

コーパスを使ってディープラーニングでモデルを作成します。

```shell
$ pipenv run create
```

## Web ページの作成

Vue.js で書かれたソースコードをビルドして Web ページを作成します。

```shell
$ pipenv run build
```

## 分類器を試す

サーバを起動します。

```shell
$ pipenv run serve
```

http://localhost:8000 にアクセスすると Web ページが表示されますので、適当な文章を入力して下さい。下部に判定結果がグラフで表示されます。

[Livedoor ニュースコーパス]: http://www.rondhuit.com/download.html#ldcc
