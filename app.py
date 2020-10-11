import streamlit as st
from bokeh.plotting import figure

import livedoor


def main():
    model = livedoor.load_model()

    st.title("ニュースの分類器")
    text = st.text_area("文章を入力してください。")

    if text:
        confidences = model.predict(text)

        sorted_labels = sorted(
            livedoor.LABELS, key=lambda x: confidences[livedoor.LABELS.index(x)]
        )

        chart = figure(y_range=sorted_labels, title="信頼性 [%]")
        chart.hbar(y=livedoor.LABELS, right=confidences)

        st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
