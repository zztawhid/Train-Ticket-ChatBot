# CMP-6059B: Intelligent Train Ticket Chatbot — Group 06

An intelligent conversational chatbot that helps users find the cheapest UK train tickets using the National Rail Online Journey Planner (OJP) SOAP API.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. Configure API credentials

Edit the `.env` file in the project root with your National Rail OJP credentials:

```
RAIL_USERNAME=your_username
RAIL_PASSWORD=your_password
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit entry point
├── chatbot/
│   ├── conversation.py       # Dialogue manager & state machine
│   ├── nlp.py                # spaCy-based NLP/intent recognition
│   ├── kb.py                 # Knowledge Base loader
│   ├── reasoning.py          # Reasoning Engine (forward-chaining)
│   ├── ticket_search.py      # National Rail OJP API (zeep/SOAP)
│   ├── station_lookup.py     # CSV station lookup + fuzzy matching
│   └── database.py           # SQLite conversation history
├── data/
│   ├── knowledge_base.json   # QnA pairs, FAQ, conversation rules
│   └── StationNameAndCode.csv
├── .env                      # API credentials (not committed)
└── requirements.txt
```

## Features

- **Natural Language Understanding** — spaCy Matcher/PhraseMatcher for intent classification, NER for entity extraction
- **Fuzzy Station Matching** — rapidfuzz resolves typos and partial station names from 2,500+ UK stations
- **Flexible Date Parsing** — dateparser handles "next Friday", "tomorrow at 10am", "before 2pm", etc.
- **Knowledge Base + Reasoning Engine** — JSON-based KB with forward-chaining inference for conversation flow
- **SOAP API Integration** — zeep client calls National Rail OJP RealtimeJourneyPlan to find cheapest fares
- **Conversation History** — SQLite database logs all messages by session
- **Professional UI** — Streamlit chat interface with travel-themed styling
