import streamlit as st
import pickle
import pandas as pd
import requests
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

params = st.query_params
# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#0E1117;
}

/* All text */
h1, h2, h3, p, label, div{
    color:white;
}

/* Selectbox */
div[data-baseweb="select"] > div{
    background-color:#1E1E1E !important;
    color:white !important;
    border:1px solid #444 !important;
    border-radius:10px;
}
div[data-baseweb="select"] input{
    color:white !important;
    -webkit-text-fill-color:white !important;
}

/* Dropdown popup */
ul{
    background-color:#1E1E1E !important;
    color:white !important;
}

/* Dropdown items */
li{
    background-color:#1E1E1E !important;
    color:white !important;
}

/* Hover effect */
li:hover{
    background-color:#333333 !important;
}

/* Button */
.stButton > button{
    background-color:#E50914;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-weight:bold;
    width:100%;
}

/* Images */
img{
    border-radius:12px;
}
/* Whole app */
.stApp{
    background-color:#0E1117;
}

/* Remove top white header */
header{
    background-color:#0E1117 !important;
}

/* Toolbar area */
[data-testid="stHeader"]{
    background-color:#0E1117 !important;
}

/* Main container */
[data-testid="stAppViewContainer"]{
    background-color:#0E1117 !important;
}

/* Sidebar (future-proof) */
[data-testid="stSidebar"]{
    background-color:#0E1117 !important;
}


</style>
""", unsafe_allow_html=True)

# Fetch movie poster
def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=21e265267aa9ce21e1d5cc19b2eddb0f'
    )

    data = response.json()

    poster_path = data['poster_path']

    full_path = "https://image.tmdb.org/t/p/w500" + poster_path

    return full_path

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=21e265267aa9ce21e1d5cc19b2eddb0f"
    return requests.get(url).json()

def fetch_movie_credits(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=21e265267aa9ce21e1d5cc19b2eddb0f"
    return requests.get(url).json()

# Recommendation function
def recommend(movie):
    movie_index = movies[
        movies['title'] == movie
        ].index.values[0]

    movie_index = movies.index.get_loc(
        movie_index
    )

    if movie_index >= cosine_sim.shape[0]:
        st.error(
            "movies_dict.pkl and similarity.pkl are mismatched."
        )

        return [], [], []

    distances = cosine_sim[
        movie_index
    ]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:8]

    recommended_movies = []
    recommended_posters = []
    ids = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        ids.append(movie_id)
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

        recommended_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movies, recommended_posters, ids


# Load data
movies_dict = pickle.load(
    open("movies_dict.pkl", "rb")
)

movies = pd.DataFrame(
    movies_dict
)

cosine_sim = pickle.load(
    open("similarity.pkl", "rb")
)

sentiment_model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

ps = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)


def clean_text(text):

    text = re.sub(
        "<.*?>",
        "",
        text
    )

    text = re.sub(
        "[^a-zA-Z]",
        " ",
        text
    )

    text = text.lower()

    words = text.split()


    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)

def predict_sentiment(review):

    cleaned_review = clean_text(
        review
    )

    vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = sentiment_model.predict(
        vector
    )[0]


    if prediction == 1:
        return "Positive"

    return "Negative"

def fetch_movie_reviews(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key=21e265267aa9ce21e1d5cc19b2eddb0f"

    data = requests.get(
        url
    ).json()

    return data["results"]

if "movie_id" in params:

    movie_id = params["movie_id"]

    details = fetch_movie_details(movie_id)

    credits = fetch_movie_credits(movie_id)


    poster = fetch_poster(movie_id)


    genres = ", ".join(
        [g["name"] for g in details["genres"]]
    )


    director = ""

    for person in credits["crew"]:

        if person["job"] == "Director":

            director = person["name"]

            break


    # Header section
    col1, col2 = st.columns([1,3])


    with col1:

        st.image(
            poster,
            use_container_width=True
        )


    with col2:

        st.markdown(
            f"""
            <h1>{details['title']}</h1>

            <p>
            {details['release_date']}
            &nbsp;&nbsp;
            {genres}
            &nbsp;&nbsp;
            {details['runtime']} mins
            </p>

            <h3>
            ⭐ {details['vote_average']}
            </h3>

            <h3>Overview</h3>

            <p>
            {details['overview']}
            </p>

            <h3>Director</h3>

            <p>
            {director}
            </p>

            """,
            unsafe_allow_html=True
        )


    st.markdown("---")

    # Cast section
    st.subheader("Starcast")

    valid_cast = []

    for actor in credits["cast"][:50]:

        if actor["profile_path"]:
            valid_cast.append(
                actor
            )

    if len(valid_cast) > 0:

        cast_cols = st.columns(
            len(valid_cast[:6])
        )

        for i, actor in enumerate(
                valid_cast[:6]
        ):
            image_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + actor["profile_path"]
            )

            with cast_cols[i]:
                st.image(
                    image_url,
                    width=150
                )

                st.markdown(
                    f"""
                    <p style='text-align:center; color:white;'>
                        {actor['name']}
                    </p>
                    """,
                    unsafe_allow_html=True
                )


    else:

        st.info(
            "No cast photos available."
        )

    st.markdown("---")

    reviews = fetch_movie_reviews(
        movie_id
    )

    with st.expander(
            "📊 Review Sentiment Analysis"
    ):

        if len(reviews) == 0:

            st.info(
                "No reviews available for this movie."
            )


        else:

            for review in reviews[:5]:
                content = review["content"]

                sentiment = predict_sentiment(
                    content
                )

                st.markdown(
                    f"### {sentiment}"
                )

                st.write(
                    content[:500]
                )


    st.markdown("---")

    st.subheader("Recommendations")

    movie_title = details["title"]

    if movie_title in movies["title"].values:

        rec_names, rec_posters, rec_ids = recommend(
            movie_title
        )

        rec_cols = st.columns(7)

        for i, col in enumerate(
                rec_cols
        ):
            with col:
                st.markdown(
                    f"""
                    <a href="?movie_id={rec_ids[i]}" target="_self">
                        <img src="{rec_posters[i]}" width="180">
                    </a>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"**{rec_names[i]}**"
                )

    if st.button("← Back"):
        st.query_params.clear()
        st.rerun()


    st.stop()
# UI
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommender System</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):

    with st.spinner("Finding similar movies..."):
        names, posters, ids = recommend(
            selected_movie_name
        )

    cols = st.columns(7)

    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <a href="?movie_id={ids[i]}" target="_self">
                    <img src="{posters[i]}" width="180">
                </a>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"**{names[i]}**"
            )

