from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
import PyPDF2
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
import google.genai as genai
from datetime import timedelta
import psycopg2
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "https://urbanrozgar.web.app",
                "https://sb8h7z34-5173.inc1.devtunnels.ms",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Accept"],
            "supports_credentials": True,  # Add this line
        }
    },
    supports_credentials=True,  # Add this line at the top level too
)

DB_NAME = os.environ.get("DB_NAME", "effihire")
DB_USER = os.environ.get("DB_USER", "admin")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PDF_PATH = os.environ.get("PDF_PATH", "./EffiHire_document.pdf")
INDEX_NAME = "pdf_index"

es_client = Elasticsearch(
    ["http://elastic:password@localhost:9200"],  # Use list format
    verify_certs=False,
    http_compress=True,
)
client = genai.Client(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def pdf_to_documents(pdf_path):
    """Extracts text from a PDF and returns a list of documents."""
    documents = []
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    documents.append(text)
        return documents
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return []
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []


def index_documents(es_client, index_name, documents):
    """Indexes documents in Elasticsearch."""
    if not es_client.indices.exists(index=index_name):
        try:
            es_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "cosine",
                            },
                        }
                    }
                },
                ignore=400,
            )
            print(f"Created index '{index_name}'")
        except Exception as e:
            print(f"Error creating index: {e}")
            return

    for i, doc in enumerate(documents):
        try:
            embedding = embedding_model.encode(doc)
            es_client.index(
                index=index_name,
                id=i,
                document={"text": doc, "embedding": embedding.tolist()},
                refresh="wait_for",
            )
        except Exception as e:
            print(f"Error indexing document {i}: {e}")


def search_documents(es_client, index_name, query, top_k=3):
    """Searches for similar documents in Elasticsearch using kNN."""
    try:
        query_embedding = embedding_model.encode(query)
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
            "_source": ["text"],
        }
        response = es_client.search(index=index_name, body=search_body)
        if response and "hits" in response and "hits" in response["hits"]:
            return [hit["_source"]["text"] for hit in response["hits"]["hits"]]
        else:
            print("Warning: No hits found or unexpected response format.")
            return []
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def generate_answer(query, context, history):
    """Generates an answer using Gemini, considering conversation history and context."""
    # Format history for the prompt
    history_prompt = ""
    last_assistant_message = ""
    if history:
        history_prompt = "Here is the conversation history:\n"
        for turn_data in history:
            history_prompt += (
                f"User: {turn_data['query']}\nAssistant: {turn_data['answer']}\n"
            )
        history_prompt += "---\n"  # Separator
        last_assistant_message = history[-1]["answer"] if history else ""

    # Detect if the user reply is a simple affirmation
    affirmations = {"yes", "yeah", "yep", "sure", "okay", "ok", "alright", "of course"}
    is_affirmation = query.strip().lower() in affirmations

    # Add greeting detection
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    is_greeting = query.strip().lower() in greetings

    # Construct the full prompt
    if is_affirmation and last_assistant_message:
        prompt = (
            f"{history_prompt}"
            f"User's last message is an affirmation ('{query}'). "
            f"As the assistant, continue the conversation by directly providing the answer or elaboration to your previous message:\n"
            f'"{last_assistant_message}"\n'
            f"Do not repeat the user's affirmation or the previous question. "
            f"Do not repeat greetings like 'Hey there! Sap here' unless this is the very first message. "
            f"Start directly with the answer, keep it natural and conversational, and encourage further questions.\n"
            f"Context Documents:\n{context}\n---\n"
        )
    elif is_greeting:
        prompt = (
            f"{history_prompt}"
            "You are EffiHire's assistant. "
            "The user has just greeted you. Respond warmly but without saying 'Hey there! Sap here'. "
            "Keep it short and ask how you can help them today. "
            f"Context Documents:\n{context}\n---\nCurrent Question: {query}"
        )
    else:
        prompt = (
            f"{history_prompt}"
            "You are a helpful assistant of EffiHire company. "
            "Only use the greeting 'Hey there! Sap here' for the very first message of a new conversation (when there is no history). "
            "For all other messages, including greetings, do not use this phrase. "
            "Answer in a precise(2 to 3 sentences), friendly, and creative way. "
            "If the context doesn't directly answer the question, use the conversation history to provide the best possible answer. "
            "If you cannot answer based on the provided information, give an appropriate response and try to divert the context towards EffiHire or gig economy. "
            "Make the answers human like, and interesting. It should appear like an actual human is talking and not a bot."
            "Give proper responses to usual greetings like Hi, Hello, Good Morning, but do not repeat greetings in every message.\n"
            f"Use the following context documents to answer the current question.\n\n"
            f"Context Documents:\n{context}\n---\nCurrent Question: {query}"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=[prompt]
        )
        if response and hasattr(response, "text"):
            answer_text = response.text
        elif response and hasattr(response, "candidates") and response.candidates:
            try:
                answer_text = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError, TypeError):
                answer_text = "NA (Error parsing complex response)"
        else:
            answer_text = "NA (Error: Empty or unexpected response structure)"

        # Fix bold text formatting
        # First, ensure all bold sections are properly paired
        parts = answer_text.split("<b>")
        fixed_text = parts[0]  # First part (before any bold)
        for i in range(1, len(parts)):
            part = parts[i]
            if "</b>" not in part:
                # Add closing tag at the end of this part
                fixed_text += f"<b>{part}</b>"
            else:
                # Keep existing structure
                fixed_text += f"<b>{part}"

        # Remove any remaining asterisks
        fixed_text = fixed_text.replace("*", "")

        return fixed_text
    except Exception as e:
        return f"NA (Error during generation: {e})"


def get_history_from_db(session_id):
    """Retrieves chat history for a given session_id from the database."""
    history = []
    conn = get_db_connection()
    if not conn:
        return history

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_query, assistant_answer
                FROM chat_history
                WHERE session_id = %s
                ORDER BY turn ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            for row in rows:
                history.append({"query": row[0], "answer": row[1]})
    except psycopg2.Error as e:
        print(f"Error fetching history from DB: {e}")
    finally:
        if conn:
            conn.close()
    return history


def save_history_to_db(session_id, turn, query, answer):
    """Saves a new chat turn to the database."""
    conn = get_db_connection()
    if not conn:
        print("Error: Could not save history, DB connection failed.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (session_id, turn, user_query, assistant_answer)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, turn, query, answer),
            )
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error saving history to DB: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def delete_history_from_db(session_id):
    """Deletes all chat history for a given session_id."""
    conn = get_db_connection()
    if not conn:
        print("Error: Could not delete history, DB connection failed.")
        return False

    deleted = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_history WHERE session_id = %s",
                (session_id,),
            )
        conn.commit()
        deleted = True
        print(f"Deleted history for session_id: {session_id}")
    except psycopg2.Error as e:
        print(f"Error deleting history from DB: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return deleted


def get_support_chat_history(ticket_id):
    """Fetch chat history for a support ticket."""
    conn = get_db_connection()
    history = []
    if not conn:
        return history
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sender, message, created_at FROM support_chat WHERE ticket_id = %s ORDER BY created_at ASC",
                (ticket_id,),
            )
            rows = cur.fetchall()
            for row in rows:
                history.append(
                    {
                        "sender": row[0],
                        "message": row[1],
                        "created_at": row[2].isoformat(),
                    }
                )
    except Exception as e:
        print(f"Error fetching support chat: {e}")
    finally:
        conn.close()
    return history


def save_support_chat_message(ticket_id, sender, message):
    """Save a message to the support chat."""
    conn = get_db_connection()
    if not conn:
        print("Error: Could not save support chat message, DB connection failed.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO support_chat (ticket_id, sender, message) VALUES (%s, %s, %s)",
                (ticket_id, sender, message),
            )
        conn.commit()
    except Exception as e:
        print(f"Error saving support chat message: {e}")
        if conn:
            conn.rollback()
    finally:
        conn.close()


@app.route("/query", methods=["POST"])
def query_endpoint():
    """Endpoint to receive a question, consider history, and return an answer."""
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id")  # Get session ID from request body

    # Validate session_id
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Validate query
    if not query:
        return jsonify({"answer": "Please provide a query."}), 400

    print(f"Processing query for session: {session_id}")  # Debug log

    # Retrieve History from Database
    history = get_history_from_db(session_id)
    current_turn = len(history) + 1

    # Search Documents (RAG part)
    context_docs = search_documents(es_client, INDEX_NAME, query)
    context = "\n---\n".join(context_docs)

    # Generate Answer
    answer = generate_answer(query, context, history)

    # Save New Turn to Database (if valid answer)
    normalized_answer = answer.strip() if isinstance(answer, str) else ""
    is_error_answer = normalized_answer.startswith("NA (Error")

    if normalized_answer != "NA" and not is_error_answer:
        save_history_to_db(session_id, current_turn, query, answer)
    else:
        print(
            f"Skipping saving history for session {session_id} due to NA/Error answer."
        )

    return jsonify({"answer": answer, "session_id": session_id})


@app.route("/clear_history", methods=["POST"])
def clear_history_endpoint():
    """Endpoint to clear the conversation history from the database."""
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    if delete_history_from_db(session_id):
        return jsonify({"message": "History cleared successfully."})
    else:
        return jsonify({"message": "Failed to clear history from database."}), 500


# Remove the session-related endpoints as they're no longer needed
# @app.route("/init_session", methods=["GET"])
# @app.route("/check_session", methods=["GET"])
# @app.before_request

# Simplify CORS since we don't need credentials anymore
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "https://urbanrozgar.web.app",
                "https://sb8h7z34-5173.inc1.devtunnels.ms",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Accept"],
            "supports_credentials": True,  # Add this line
        }
    },
    supports_credentials=True,  # Add this line at the top level too
)


@app.route("/api/support_chat/<int:ticket_id>", methods=["GET"])
def get_support_chat(ticket_id):
    """Get chat history for a support ticket."""
    history = get_support_chat_history(ticket_id)
    return jsonify({"history": history})


@app.route("/api/support_chat/<int:ticket_id>", methods=["POST"])
def post_support_chat(ticket_id):
    """Send a message to the support chat and get Gemini's response."""
    data = request.get_json()
    user_message = data.get("message")
    sender = data.get("sender", "client")  # 'client', 'admin', or 'support'

    if not user_message:
        return jsonify({"error": "Message required"}), 400

    # Save user/admin/support message
    save_support_chat_message(ticket_id, sender, user_message)

    # Check if manual mode is enabled
    conn = get_db_connection()
    manual_mode = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT manual_mode FROM support_tickets WHERE id = %s", (ticket_id,)
            )
            row = cur.fetchone()
            if row:
                manual_mode = row[0]
    finally:
        conn.close()

    # If admin/support is replying, or manual mode is on, don't generate Gemini reply
    if sender in ("admin", "support") or manual_mode:
        return jsonify({"answer": None})

    # Prepare history for Gemini
    history = get_support_chat_history(ticket_id)
    formatted_history = ""
    for turn in history:
        formatted_history += f"{turn['sender'].capitalize()}: {turn['message']}\n"
    prompt = (
        "You are EffiHire Support Assistant. "
        "Reply concisely in 1 or 2 sentences. "
        "Continue this support conversation:\n"
        f"{formatted_history}\n"
        f"{sender.capitalize()}: {user_message}\n"
        "Assistant:"
    )

    # Generate Gemini response
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=[prompt]
        )
        if response and hasattr(response, "text"):
            answer = response.text
        elif response and hasattr(response, "candidates") and response.candidates:
            answer = response.candidates[0].content.parts[0].text
        else:
            answer = "Sorry, I couldn't generate a response."
    except Exception as e:
        answer = f"Error generating response: {e}"

    # Remove asterisks from the answer
    answer = answer.replace("*", "")

    # Save Gemini/bot response
    save_support_chat_message(ticket_id, "bot", answer)

    return jsonify({"answer": answer})


@app.route("/api/support_chat/<int:ticket_id>/manual_mode", methods=["GET"])
@cross_origin(
    origins=["http://localhost:5173", "https://urbanrozgar.web.app"],
    supports_credentials=True,
)
def get_manual_mode(ticket_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT manual_mode FROM support_tickets WHERE id = %s", (ticket_id,)
            )
            row = cur.fetchone()
            if row:
                return jsonify({"manual_mode": row[0]})
            else:
                return jsonify({"manual_mode": False})
    finally:
        conn.close()


@app.route("/api/support_chat/<int:ticket_id>/manual_mode", methods=["POST"])
@cross_origin(
    origins=["http://localhost:5173", "https://urbanrozgar.web.app"],
    supports_credentials=True,
)
def set_manual_mode(ticket_id):
    """Set manual mode for a support ticket."""
    data = request.get_json()
    manual_mode = data.get("manual_mode", False)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE support_tickets SET manual_mode = %s WHERE id = %s",
                (manual_mode, ticket_id),
            )
        conn.commit()
        return jsonify({"success": True, "manual_mode": manual_mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/support_ticket/<int:ticket_id>/close", methods=["POST"])
def close_support_ticket(ticket_id):
    """Close a support ticket and add a final message."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Set status to closed
            cur.execute(
                "UPDATE support_tickets SET status = 'closed' WHERE id = %s",
                (ticket_id,),
            )
            # Add final message
            cur.execute(
                "INSERT INTO support_chat (ticket_id, sender, message) VALUES (%s, %s, %s)",
                (
                    ticket_id,
                    "support",
                    "This ticket is being closed by support. You can view the chat history, but you cannot reply further.",
                ),
            )
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/init_session", methods=["GET"])
def init_session():
    """Initialize a new session and return session_id."""
    new_session_id = str(uuid.uuid4())
    return jsonify({"session_id": new_session_id, "status": "Session initialized"})


if __name__ == "__main__":
    try:
        if (
            not es_client.indices.exists(index=INDEX_NAME)
            or es_client.count(index=INDEX_NAME)["count"] == 0
        ):
            print(f"Index '{INDEX_NAME}' not found or empty. Indexing PDF...")
            documents = pdf_to_documents(PDF_PATH)
            if documents:
                index_documents(es_client, INDEX_NAME, documents)
                print("PDF indexed successfully.")
            else:
                print("Failed to load documents from PDF. Cannot index.")
        else:
            print(f"Index '{INDEX_NAME}' already exists and contains documents.")
    except Exception as e:
        print(f"Error during initial index check/creation: {e}")

    # Use PORT environment variable for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
