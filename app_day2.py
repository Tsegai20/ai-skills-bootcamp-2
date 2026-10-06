import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="News Classifier — Fine-tuned BERT",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Classifier")
st.write("Fine-tuned BERT model classifying news into World, Sports, Business, and Sci/Tech")
st.divider()

label_names = ["World", "Sports", "Business", "Sci/Tech"]
label_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./my_news_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("./my_news_classifier")
    model.eval()
    return tokenizer, model

with st.spinner("Loading fine-tuned BERT model..."):
    tokenizer, model = load_model()

st.success("Model loaded successfully")

# Input section
st.subheader("Enter a news headline or sentence")

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Text to classify",
        placeholder="e.g. NASA launches new rocket to explore Mars...",
        height=120
    )

with col2:
    st.markdown("**Try these examples:**")
    examples = [
        "NASA launches new rocket to Mars",
        "Manchester United beats Chelsea 3-1",
        "Federal Reserve raises interest rates",
        "Apple releases new iPhone with AI chip",
        "UAV uses deep learning for navigation",
        "Federated learning preserves privacy"
    ]
    for example in examples:
        if st.button(example, key=example):
            text_input = example

if st.button("Classify", type="primary") or text_input:
    if text_input and text_input.strip():
        with st.spinner("Classifying..."):
            inputs = tokenizer(
                text_input,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        st.divider()

        # Result
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.metric("Predicted Category", label_names[predicted_class])
            st.metric("Confidence", f"{confidence:.2%}")

            if confidence >= 0.95:
                st.success("Very high confidence")
            elif confidence >= 0.80:
                st.info("High confidence")
            else:
                st.warning("Moderate confidence")

        with col_b:
            # Bar chart of all probabilities
            probs = [p.item() for p in probabilities]

            fig = px.bar(
                x=label_names,
                y=probs,
                color=label_names,
                color_discrete_sequence=label_colors,
                title="Prediction Probabilities",
                labels={"x": "Category", "y": "Probability"}
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Probability table
        st.subheader("Detailed Scores")
        prob_df = pd.DataFrame({
            "Category": label_names,
            "Probability": [f"{p.item():.2%}" for p in probabilities],
            "Score": [p.item() for p in probabilities]
        }).sort_values("Score", ascending=False)

        prob_df["Rank"] = range(1, 5)
        st.dataframe(
            prob_df[["Rank", "Category", "Probability"]],
            use_container_width=True,
            hide_index=True
        )

# Batch classification section
st.divider()
st.subheader("Batch Classification")
st.write("Classify multiple headlines at once")

batch_text = st.text_area(
    "Enter multiple headlines — one per line",
    placeholder="NASA launches rocket to Mars\nManchester United wins championship\nFed raises interest rates",
    height=150
)

if st.button("Classify All", type="secondary"):
    if batch_text.strip():
        headlines = [h.strip() for h in batch_text.strip().split("\n") if h.strip()]

        results = []
        with st.spinner(f"Classifying {len(headlines)} headlines..."):
            for headline in headlines:
                inputs = tokenizer(
                    headline,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_class = torch.argmax(probs).item()
                conf = probs[pred_class].item()

                results.append({
                    "Headline": headline[:60] + "..." if len(headline) > 60 else headline,
                    "Category": label_names[pred_class],
                    "Confidence": f"{conf:.2%}"
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Distribution chart
        category_counts = results_df["Category"].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Category Distribution",
            color_discrete_sequence=label_colors
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()
st.caption("Fine-tuned BERT model — trained on AG News dataset | Built by Tsegai Yhdego — PhD Industrial Engineering | AI/ML Researcher")
