import streamlit as st

st.set_page_config(
    page_title = "Home"
)

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "ML Showcase";
                margin-left: 20px;
                margin-top: 20px;
                margin-bottom: 60px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

#st.markdown("<h1 style='text-align: center; color: white;'>ML Showcase</h1>", unsafe_allow_html=True)
st.image("assets/Untitled Project.jpg")
st.markdown("""---""")
st.subheader("What is this app about?")
st.write("This app shows you a few machine learning algorithms, how they work and where they can be used. You will also be able to edit the model parameters how you wish and see how the accuracy varies as a result.")
st.write("This app will also allow you to use different datasets and visualise the data.")
st.markdown("<p style='color: green; font-weight: bold;'>To start, click on one of the pages on the sidebar to the left.</p>", unsafe_allow_html=True)
#st.write("To start, click on one of the pages on the sidebar to the left.")



st.markdown("""---""")
st.subheader("Some Extra Help On Learning ML:")
st.write("- [YT Playlist 1](https://www.youtube.com/playlist?list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw)")
st.write("- [YT Playlist 2](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)")
st.write("- [YT Playlist 3](https://www.youtube.com/playlist?list=PL9ooVrP1hQOHUfd-g8GUpKI3hHOwM_9Dn)")
st.markdown("""---""")
st.subheader("Good Luck!")
st.write("[GitHub](https://github.com/miranamer/)")

st.sidebar.success("Select A Page")