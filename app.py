import streamlit as st
from bokeh.plotting import figure

import dataset
from training import dnn


def main():
    st.title("ニュースの分類器")

    mecab = dataset.MeCabTokenizer()
    tokenizer = dataset.get_tokenizer()
    model = dnn.load_model("model.h5")

    text = st.text_area("文章を入力してください。")

    if text:
        texts = [mecab.tokenize(text)]
        tfidf = tokenizer.texts_to_matrix(texts, mode="tfidf")
        confidences = model.predict(tfidf)[0]

        labels = dataset.LABELS
        sorted_labels = sorted(labels, key=lambda x: confidences[labels.index(x)])

        chart = figure(y_range=sorted_labels, title="信頼性 [%]")
        chart.hbar(y=labels, right=confidences)

        st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
