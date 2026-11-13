# ✍️ AI-Powered LinkedIn Post Generator

An AI-powered LinkedIn Post Generator that creates engaging and professional LinkedIn posts using multiple LLMs, web search, and automated content review.

Built with **Python, LangGraph, Streamlit, NVIDIA Nemotron, Groq, and Tavily Search**.

## 🚀 Project Overview

Creating engaging LinkedIn content takes time and effort. This project simplifies the process by using AI to generate, review, and improve LinkedIn posts automatically.

The application generates a post based on a user-provided topic, evaluates it against predefined content criteria, and improves rejected drafts through an iterative workflow.

## ✨ Features

- Generate professional LinkedIn posts from a given topic.
- Use Tavily Search to retrieve current information when required.
- Review generated posts automatically using an LLM.
- Improve rejected drafts using reviewer feedback.
- Manage the complete workflow using LangGraph.
- Interactive user interface built with Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangGraph | Workflow orchestration |
| NVIDIA Nemotron | LinkedIn post generation |
| Groq GPT-OSS | Content review |
| Tavily Search | Web search |
| python-dotenv | Environment variable management |

## 🔄 Workflow

1. User enters a LinkedIn post topic.
2. NVIDIA Nemotron generates the initial draft.
3. Tavily Search retrieves information when required.
4. The generated draft is sent to the reviewer.
5. Groq evaluates the post against predefined criteria.
6. If rejected, reviewer feedback is sent back to the writer.
7. The process repeats until approval or the maximum number of attempts is reached.
8. The final post and review feedback are displayed in Streamlit.

## 📁 Project Structure

```text
LinkedIn_Post_Generator/
│
├── iterativiUI.py
├── .env.example
├── .gitignore
├── iterativitool.py
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd LinkedIn_Post_Generator
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the root directory and add your API keys:

```env
NVIDIA_API_KEY=your_nvidia_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Get API keys from:

- NVIDIA: https://build.nvidia.com/
- Groq: https://console.groq.com/
- Tavily: https://tavily.com/

**Important:** Never upload your `.env` file or expose your API keys on GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📸 Application Preview

Add your application screenshots here:

```markdown
![Application UI](screenshots/home.png)
```

## 🎯 Learning Outcomes

Through this project, I explored:

- Building AI applications using LangGraph.
- Connecting multiple LLM providers.
- Integrating external tools with LLMs.
- Implementing conditional routing and iterative workflows.
- Building interactive AI applications using Streamlit.

## 🔮 Future Scope

- Further improvements to content quality and review accuracy.
- Additional writing styles and customization options.

## 👨‍💻 Author

**Prince Gupta**

GitHub: https://github.com/princegupta4171

---

If you find this project interesting, feel free to ⭐ the repository!

Suggestions and feedback are always welcome.
