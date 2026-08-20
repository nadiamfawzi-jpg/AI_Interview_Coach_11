import os
import tempfile

import pandas as pd
import streamlit as st

from interview_utils import check_answer
from video_utils import annotate_video, annotate_pose


st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eefbff 52%, #fff8ed 100%);
}
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102a43, #153e75);
}
[data-testid="stSidebar"] * {
    color: white;
}
.hero {
    padding: 34px 38px;
    border-radius: 24px;
    background: linear-gradient(120deg, #102a43, #2563eb 58%, #06b6d4);
    color: white;
    box-shadow: 0 18px 45px #102a4330;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 2.7rem;
    margin: 0 0 8px;
}
.hero p {
    color: #e6f4ff;
    font-size: 1.05rem;
    margin: 0;
    max-width: 800px;
}
.badge {
    display: inline-block;
    padding: 7px 12px;
    margin-bottom: 12px;
    border: 1px solid #ffffff40;
    border-radius: 100px;
    background: #ffffff20;
    font-size: .8rem;
    font-weight: 700;
}
.question {
    padding: 22px;
    margin: 8px 0 18px;
    border-left: 7px solid #2563eb;
    border-radius: 16px;
    background: linear-gradient(120deg, #eff6ff, white);
    color: #102a43;
    font-size: 1.15rem;
    font-weight: 750;
}
.notice {
    padding: 15px 17px;
    margin-bottom: 18px;
    border: 1px solid #fed7aa;
    border-radius: 13px;
    background: #fff7ed;
    color: #7c2d12;
}
.idea-yes {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 100px;
    background: #dcfce7;
    color: #166534;
    font-weight: 700;
}
.idea-no {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 100px;
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
}
.stButton > button, .stDownloadButton > button {
    min-height: 44px;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    font-weight: 800;
}
div[data-testid="stMetric"] {
    padding: 15px;
    border: 1px solid #dbeafe;
    border-radius: 15px;
    background: #ffffffdd;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="badge">SMART PRACTICE • HONEST FEEDBACK</div>
    <h1>🎯 AI Interview Coach</h1>
    <p>Practise interview questions, check whether your answer covers important ideas,
    and create a YOLO-annotated copy of your recorded interview video.</p>
</div>
""", unsafe_allow_html=True)

app_folder = os.path.dirname(os.path.abspath(__file__))
questions_df = pd.read_csv(os.path.join(app_folder, "questions.csv"))

with st.sidebar:
    st.title("⚙️ Practice Setup")
    field = st.selectbox("Choose your field", questions_df["Field"].unique())

    field_questions = questions_df[
        questions_df["Field"] == field
    ].reset_index(drop=True)

    question_number = st.selectbox(
        "Choose a question",
        range(len(field_questions)),
        format_func=lambda number: "Question " + str(number + 1)
    )

    st.divider()
    st.markdown("**Methods used**")
    st.caption("pandas • keyword checks • YOLO • OpenCV")
    st.divider()
    st.caption("Practice support only — not a recruitment tool.")

selected = field_questions.iloc[question_number]

answer_tab, video_tab, guide_tab = st.tabs([
    "💬 Answer Coach",
    "🎥 Video Lab",
    "📘 How It Works"
])

with answer_tab:
    st.subheader(field + " Interview Practice")
    st.markdown(
        '<div class="question">❓ ' + selected["Question"] + '</div>',
        unsafe_allow_html=True
    )

    answer = st.text_area(
        "Your answer",
        height=190,
        placeholder="Write your answer as if you are speaking to the interviewer..."
    )

    if st.button("✨ Review My Answer", type="primary"):
        if answer.strip() == "":
            st.warning("Please write your answer first.")
        else:
            st.session_state["answer_result"] = check_answer(
                answer,
                selected["Keywords"]
            )

    if "answer_result" in st.session_state:
        result = st.session_state["answer_result"]

        st.markdown("### Your content checklist")
        column1, column2, column3 = st.columns(3)
        column1.metric("Words", result["word_count"])
        column2.metric("Ideas included", len(result["matched_words"]))
        column3.metric("Ideas to consider", len(result["missing_words"]))

        st.markdown("#### ✅ Ideas found")
        if result["matched_words"]:
            found_html = "".join(
                '<span class="idea-yes">' + word + '</span>'
                for word in result["matched_words"]
            )
            st.markdown(found_html, unsafe_allow_html=True)
        else:
            st.info("No supplied key ideas were found literally yet.")

        st.markdown("#### 💡 Ideas you may add")
        if result["missing_words"]:
            missing_html = "".join(
                '<span class="idea-no">' + word + '</span>'
                for word in result["missing_words"]
            )
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.success("All supplied key ideas appear in your answer.")

        with st.expander("View the sample answer"):
            st.write(selected["Ideal_Answer"])

        st.caption(
            "This is a literal keyword checklist, not a trained NLP quality score. "
            "It may miss synonyms and context."
        )

with video_tab:
    st.subheader("Recorded Interview Video Lab")
    st.markdown("""
    <div class="notice"><b>Important:</b> The video model only draws visible
    objects or body keypoints. It does not detect emotions, nervousness,
    confidence, honesty, personality, eye contact, or interview readiness.</div>
    """, unsafe_allow_html=True)

    video_method = st.radio(
        "Choose annotation type",
        ["Object detection", "Pose keypoints"],
        horizontal=True
    )

    uploaded_video = st.file_uploader(
        "Upload your recorded interview",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🎬 Create Annotated Video", type="primary"):
            suffix = "." + uploaded_video.name.split(".")[-1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as input_file:
                input_file.write(uploaded_video.getbuffer())
                input_path = input_file.name

            output_path = os.path.join(
                tempfile.gettempdir(),
                "annotated_interview.mp4"
            )

            with st.spinner("YOLO is annotating the video..."):
                if video_method == "Pose keypoints":
                    annotate_pose(input_path, output_path)
                else:
                    annotate_video(input_path, output_path)

            st.success("Annotated video completed.")
            st.video(output_path)

            with open(output_path, "rb") as video_file:
                st.download_button(
                    "⬇️ Download Annotated Video",
                    video_file,
                    file_name="annotated_interview.mp4",
                    mime="video/mp4"
                )

with guide_tab:
    st.subheader("What this version genuinely does")
    st.markdown("""
    - Loads the supplied question bank with pandas.
    - Checks for manually supplied key ideas using simple Python string checks.
    - Shows a sample answer for self-review.
    - Uses tutor-demonstrated YOLO and OpenCV patterns to annotate a video.

    **Not included:** TF-IDF, MediaPipe, Streamlit WebRTC, speech analysis,
    semantic scoring, emotion recognition, nervousness detection, candidate
    ranking, or an overall interview score.
    """)

