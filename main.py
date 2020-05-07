import streamlit as st
from bokeh.plotting import figure

from news_classifier import preprocessing, livedoor, dnn


def main():
    st.title("ニュースの分類器")

    mecab = preprocessing.MeCabTokenizer()
    tokenizer = livedoor.get_tokenizer()
    model = dnn.load_model("model.h5")

    text = st.text_area("文章を入力してください。")

    if text:
        texts = [mecab.tokenize(text)]
        tfidf = tokenizer.texts_to_matrix(texts, mode="tfidf")
        confidences = model.predict(tfidf)

        categories = list(livedoor.CATEGORIES.values())
        sorted_categories = sorted(categories, key=lambda x: confidences[0][categories.index(x)])

        chart = figure(y_range=sorted_categories, title="信頼性 [%]")
        chart.hbar(y=categories, right=confidences[0])

        st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
