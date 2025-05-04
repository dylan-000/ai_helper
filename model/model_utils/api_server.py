from flask import Flask, jsonify, request, abort
from processor import QueryProcessor
from model_test import setup_vector_store, print_result

app = Flask(__name__)

# Startup our query processor and load the collection to be used with it
PERSIST_PATH = "model/data/chroma_db"
COLLECTION = setup_vector_store(PERSIST_PATH)
QUERY_PROC = QueryProcessor(
    vector_collection=COLLECTION
)

@app.route("/ask", methods=["POST"])
def ask():

    """
    Expects JSON like: {"question": "How do I get a cloud in a bottle?"}
    """
    try:
        data = request.get_json(silent=True)
        if not data or "question" not in data:
            abort(400, description="JSON must contain a 'question' field")

        question = data["question"]
        raw = QUERY_PROC.process_query(question)
        response = raw['response']
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")     

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)