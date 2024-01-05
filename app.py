from bokeh.plotting import figure
import streamlit as st

import livedoor


def main():
    model = livedoor.load_model()

    st.title("ニュース記事の分類器")

    with st.form(key="form"):
        text = st.text_area(
            "記事を貼り付けると、どのジャンルの記事なのかを推定します。",
            height=400,
        )
        submit_button = st.form_submit_button(label="推定する")

    if submit_button or text:
        st.caption("")
        st.markdown("## 結果")
        st.caption("")

        categories = model.predict(text)
        st.write(categories)


if __name__ == "__main__":
    main()
