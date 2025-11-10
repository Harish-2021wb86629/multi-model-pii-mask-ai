import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from mistralai import Mistral # Correct import based on your documentation

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATIONS ---
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")

# Initialize Mistral Client correctly
client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': 'No messages provided'}), 400

    incoming_messages = data['messages']

    try:
        # --- TRANSFORMATION LOGIC (Still Required) ---
        # This part correctly converts the format from your main app
        messages_for_mistral = []
        for msg in incoming_messages:
            role = msg['role']
            if role == 'model':
                role = 'assistant' # Mistral API uses 'assistant'

            # Extract the text content from the Gemini-style format
            text_content = msg.get('parts', [{}])[0].get('text', '')

            # Create the dictionary that the Mistral API expects
            messages_for_mistral.append(
                {"role": role, "content": text_content}
            )

        # --- CORRECT API CALL ---
        # Use the client.chat.complete method as per the documentation
        chat_response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages_for_mistral,
        )

        assistant_reply = chat_response.choices[0].message.content

        return jsonify({'response': assistant_reply})

    except Exception as e:
        print(f"Error calling Mistral API or processing data: {e}")
        return jsonify({'error': f"An unexpected error occurred in mistral-service: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
