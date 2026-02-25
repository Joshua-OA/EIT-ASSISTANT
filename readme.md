# Joshua's EIT Assistant

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key — get one at [platform.openai.com](https://platform.openai.com)

---

## Installation

**1. Clone or download the project, then navigate into the folder:**

```bash
cd "16 FEB"
```

**2. Create and activate a virtual environment:**

```bash
python -m venv mestvenv
source mestvenv/bin/activate        # Mac / Linux
mestvenv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## API Key Setup

Create a file named `.env` in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

> You can get your key from: https://platform.openai.com/api-keys

---

## Running the App

```bash
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## What You Can Do

- **Chat** — type any question and press Enter
- **Upload a PDF** — click `📄 Upload PDF` below the chat input
- **Summarize a link** — click `🔗 Link` to activate URL mode, paste the URL into the input, then press Enter

---

## Deploying to Fly.io

**1. Install the Fly CLI:**

Follow the instructions at [fly.io/docs/flyctl/install](https://fly.io/docs/flyctl/install/).

**2. Launch your app (first time only):**

```bash
fly launch
```

**3. Set your OpenAI API key as a secret:**

```bash
fly secrets set OPENAI_API_KEY=your-openai-api-key
```

**4. Deploy:**

```bash
fly deploy
```

The app will be available at `https://<your-app-name>.fly.dev`.

> **Note:** Update the `app` name in `fly.toml` to match the app name you created on Fly.io.
