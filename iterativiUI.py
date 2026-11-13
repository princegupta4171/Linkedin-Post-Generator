
import os
import streamlit as st

from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="wide"
)


# ==================================================
# 2. CUSTOM CSS - UI DESIGN ONLY
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1150px;
}

.main-heading {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.sub-heading {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #172554, #312e81);
    padding: 30px;
    border-radius: 16px;
    border: 1px solid #3730a3;
    margin-bottom: 25px;
}

.hero h2 {
    color: white;
    font-size: 27px;
    margin-bottom: 8px;
}

.hero p {
    color: #c7d2fe;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-card {
    background: #191f2b;
    border: 1px solid #30394a;
    padding: 22px;
    border-radius: 14px;
}

div.stButton > button {
    border-radius: 9px;
    height: 46px;
    font-weight: 650;
}

[data-testid="stSidebar"] {
    background-color: #151a24;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE
# ==================================================

if "final_state" not in st.session_state:
    st.session_state.final_state = None


# ==================================================
# 4. TOOLS
# ==================================================

search_tool = TavilySearch(max_results=3)

tools = [search_tool]

tool_node = ToolNode(tools)


# ==================================================
# 5. LLM MODELS
# ==================================================

writer_llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.7,
    max_tokens=1024
)

writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# ==================================================
# 6. STATE
# ==================================================

class State(TypedDict):

    topic: str

    messages: Annotated[list, add_messages]

    draft: str

    review_feedback: str

    is_approved: bool

    attempt: int


# ==================================================
# 7. WRITER SYSTEM PROMPT
# ==================================================

WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer.

Your job is to generate complete, engaging and
professional LinkedIn posts.

Never ask the user for feedback or clarification.

If reviewer feedback is provided, fix every issue.

If current information is required, use web search.

Follow these rules:

1. Strong hook in the first line.
2. One clear takeaway.
3. Short and readable paragraphs.
4. Around 150-200 words.
5. End with an engaging question or CTA.
6. Professional but human tone.
7. No hashtags.

Always return the complete LinkedIn post.
"""


# ==================================================
# 8. WRITER NODE
# ==================================================

def writer_node(state: State) -> dict:

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    previous_feedback = state.get("review_feedback", "")

    previous_draft = state.get("draft", "")

    if attempt == 1:

        user_message = f"""
        Write a complete LinkedIn post about: {topic}.

        If current information is required, search the web.

        Generate the complete post directly.
        """

    else:

        user_message = f"""
        Improve the LinkedIn post about: {topic}.

        Previous draft:
        {previous_draft}

        Reviewer feedback:
        {previous_feedback}

        Fix every issue mentioned in the feedback.

        Generate the complete improved post directly.
        """

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [
            ("human", user_message),
            response
        ],
        "attempt": attempt
    }


# ==================================================
# 9. TOOL NODE
# ==================================================

tool_node = ToolNode(tools)


# ==================================================
# 10. EXTRACT DRAFT NODE
# ==================================================

def extract_draft_node(state: State) -> dict:

    last_message = state["messages"][-1]

    draft = last_message.content

    if isinstance(draft, list):

        draft = "\n".join(
            item.get("text", "")
            for item in draft
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    if not isinstance(draft, str) or not draft.strip():

        raise ValueError(
            "Writer did not generate post content."
        )

    return {
        "draft": draft.strip()
    }


# ==================================================
# 11. REVIEWER SYSTEM PROMPT
# ==================================================

REVIEWER_SYSTEM_PROMPT = """
You are a strict LinkedIn content reviewer.

Evaluate the post on these criteria:

1. Strong hook in the first line.
2. One clear takeaway.
3. Short and readable paragraphs.
4. Around 150-200 words.
5. Engaging ending or CTA.
6. Professional but human tone.
7. No hashtags.

Respond in exactly this format:

VERDICT: APPROVED or REJECTED

FEEDBACK: One short paragraph explaining why.

Approve only if the post meets all criteria.
"""


# ==================================================
# 12. REVIEWER NODE
# ==================================================

def reviewer_node(state: State) -> dict:

    draft = state["draft"]

    prompt = f"""
    Review this LinkedIn post:

    {draft}

    Evaluate all required criteria.
    """

    response = reviewer_llm.invoke([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", prompt)
    ])

    review_text = response.content.strip()

    verdict_section = review_text.upper().split("FEEDBACK:")[0]

    is_approved = "VERDICT: APPROVED" in verdict_section

    if "FEEDBACK:" in review_text:

        feedback = review_text.split(
            "FEEDBACK:", 1
        )[1].strip()

    else:

        feedback = review_text

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ==================================================
# 13. ROUTER FUNCTIONS
# ==================================================

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):

        return "tools"

    return "extract_draft"


def should_stop_looping(state: State):

    if state["is_approved"]:

        return END

    if state["attempt"] >= 3:

        return END

    return "writer"


# ==================================================
# 14. BUILD LANGGRAPH
# ==================================================

graph = StateGraph(State)

graph.add_node("writer", writer_node)

graph.add_node("tools", tool_node)

graph.add_node("extract_draft", extract_draft_node)

graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")

graph.add_conditional_edges(
    "writer",
    should_use_tool,
    {
        "tools": "tools",
        "extract_draft": "extract_draft"
    }
)

graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)

app = graph.compile()


# ==================================================
# 15. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## ✍️ LinkedIn Generator")

    st.caption("AI-powered content creation")

    st.divider()

    st.markdown("### AI Models")

    st.markdown("""
    **Writer**

    NVIDIA Nemotron

    **Reviewer**

    Groq GPT-OSS

    **Search**

    Tavily
    """)

    st.divider()

    st.markdown("### Workflow")

    st.markdown("""
    1. Enter your topic
    2. Generate LinkedIn post
    3. AI reviews the draft
    4. Writer improves rejected drafts
    5. View final result
    """)


# ==================================================
# 16. MAIN HEADER
# ==================================================

st.markdown(
    '<div class="main-heading">✍️ LinkedIn Post Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-heading">'
    'Create engaging and professional LinkedIn posts '
    'with AI-powered writing and reviewing.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# 17. HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">

    <h2>Turn your ideas into powerful LinkedIn posts.</h2>

    <p>
        Write better content with AI.
        Automatically review and improve your posts.
    </p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# 18. TOPIC INPUT
# ==================================================

st.markdown(
    '<div class="section-title">🚀 Enter Your Topic</div>',
    unsafe_allow_html=True
)

topic = st.text_area(
    "What would you like to write about?",
    placeholder=(
        "Example: My journey in learning Generative AI..."
    ),
    height=120
)


# ==================================================
# 19. GENERATE BUTTON
# ==================================================

generate_button = st.button(
    "✨ Generate LinkedIn Post",
    type="primary",
    use_container_width=True
)


if generate_button:

    if not topic.strip():

        st.warning("Please enter a topic first.")

    else:

        initial_state = {

            "topic": topic.strip(),

            "messages": [],

            "draft": "",

            "review_feedback": "",

            "is_approved": False,

            "attempt": 0

        }

        try:

            with st.spinner(
                "Generating and reviewing your post..."
            ):

                final_state = app.invoke(
                    initial_state,
                    config={"recursion_limit": 25}
                )

            st.session_state.final_state = final_state

            st.success("LinkedIn post generated successfully!")

        except Exception as e:

            st.error(f"Error: {str(e)}")


# ==================================================
# 20. DISPLAY FINAL POST
# ==================================================

if st.session_state.final_state is not None:

    result = st.session_state.final_state

    st.divider()

    st.markdown(
        '<div class="section-title">📝 Generated LinkedIn Post</div>',
        unsafe_allow_html=True
    )

    st.text_area(
        "Final Post",
        value=result["draft"],
        height=300
    )


    # ==================================================
    # 21. REVIEW RESULTS
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Review Results</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Attempts",
            result["attempt"]
        )

    with col2:

        if result["is_approved"]:

            st.success("APPROVED")

        else:

            st.error("REJECTED")


    # ==================================================
    # 22. REVIEWER FEEDBACK
    # ==================================================

    st.markdown(
        '<div class="section-title">💬 Reviewer Feedback</div>',
        unsafe_allow_html=True
    )

    st.info(result["review_feedback"])
