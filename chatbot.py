import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# -----------------------------------
# 1. INTENTS / KNOWLEDGE BASE
# -----------------------------------

intents = [

    {
        "tag": "greeting",
        "patterns": [
            "Hi",
            "Hello",
            "Hey",
            "Good morning",
            "Good evening",
            "Is anyone there?"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Welcome! How may I assist you?"
        ]
    },

    {
        "tag": "how_are_you",
        "patterns": [
            "How are you?",
            "How's it going?",
            "Are you fine?",
            "What are you doing?"
        ],
        "responses": [
            "I'm doing great! Thanks for asking.",
            "I'm here helping customers like you.",
            "All good! How can I assist you today?"
        ]
    },

    {
        "tag": "bot_name",
        "patterns": [
            "What is your name?",
            "Who are you?",
            "Tell me your name"
        ],
        "responses": [
            "I'm your Customer Service Assistant Bot.",
            "You can call me SupportBot.",
            "I'm an AI chatbot created to help you."
        ]
    },

    {
        "tag": "creator",
        "patterns": [
            "Who created you?",
            "Who made you?",
            "Who developed you?"
        ],
        "responses": [
            "I was created using Python and Machine Learning.",
            "A developer trained me to assist users.",
            "I was built to help customers."
        ]
    },

    {
        "tag": "thanks",
        "patterns": [
            "Thank you",
            "Thanks",
            "Thanks a lot",
            "Appreciate it"
        ],
        "responses": [
            "You're welcome!",
            "Happy to help.",
            "Anytime!"
        ]
    },

    {
        "tag": "help",
        "patterns": [
            "Can you help me?",
            "I need help",
            "Help me",
            "Support needed"
        ],
        "responses": [
            "Of course! Tell me your issue.",
            "I'm here to help you.",
            "Please describe your problem."
        ]
    },

    {
        "tag": "jokes",
        "patterns": [
            "Tell me a joke",
            "Make me laugh",
            "Say something funny"
        ],
        "responses": [
            "Why did the computer go to the doctor? Because it caught a virus!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I would tell you a UDP joke, but you might not get it."
        ]
    },

    {
        "tag": "age",
        "patterns": [
            "How old are you?",
            "What is your age?",
            "When were you created?"
        ],
        "responses": [
            "Age doesn't apply to bots like me!",
            "I exist whenever the program runs.",
            "I'm always new every time you start me."
        ]
    },

    {
        "tag": "capabilities",
        "patterns": [
            "What can you do?",
            "How can you help?",
            "What are your features?"
        ],
        "responses": [
            "I can answer customer service questions.",
            "I can help with shipping, returns, and orders.",
            "I'm trained to assist users with common queries."
        ]
    },

    {
        "tag": "hours",
        "patterns": [
            "What are your hours?",
            "When are you open?",
            "Store timings",
            "What time do you close?"
        ],
        "responses": [
            "We are open from 9 AM to 6 PM, Monday to Friday."
        ]
    },

    {
        "tag": "returns",
        "patterns": [
            "How do I return an item?",
            "Return policy",
            "I want a refund",
            "Can I return this?"
        ],
        "responses": [
            "You can return items within 30 days with a valid receipt."
        ]
    },

    {
        "tag": "shipping",
        "patterns": [
            "How long does shipping take?",
            "Track my order",
            "Where is my package?",
            "Delivery time"
        ],
        "responses": [
            "Shipping usually takes 3-5 business days."
        ]
    },

    {
        "tag": "payment",
        "patterns": [
            "Payment failed",
            "Unable to pay",
            "Transaction issue",
            "Card declined"
        ],
        "responses": [
            "Please check your payment details and try again.",
            "If money was deducted, it will be refunded within 5-7 business days."
        ]
    },

    {
        "tag": "cancel_order",
        "patterns": [
            "Cancel my order",
            "I want to cancel",
            "Stop my order"
        ],
        "responses": [
            "Your order can be cancelled before shipment.",
            "Please provide your order ID for cancellation."
        ]
    },

    {
        "tag": "account_help",
        "patterns": [
            "Forgot password",
            "Can't login",
            "Reset password",
            "Login problem"
        ],
        "responses": [
            "Use the 'Forgot Password' option to reset your password.",
            "Please check your email for reset instructions."
        ]
    },

    {
        "tag": "product_availability",
        "patterns": [
            "Is this product available?",
            "In stock?",
            "Do you have this item?"
        ],
        "responses": [
            "Please share the product name to check availability.",
            "Our stock updates every hour."
        ]
    },

    {
        "tag": "discounts",
        "patterns": [
            "Any discounts?",
            "Coupon code",
            "Offers available"
        ],
        "responses": [
            "Check the Offers section on our website.",
            "New discounts are updated every weekend."
        ]
    },

    {
        "tag": "complaint",
        "patterns": [
            "I want to complain",
            "Bad service",
            "Very disappointed"
        ],
        "responses": [
            "We're sorry for the inconvenience.",
            "Please explain the issue so we can help you better."
        ]
    },

    {
        "tag": "contact_support",
        "patterns": [
            "Talk to human",
            "Customer support",
            "Need agent"
        ],
        "responses": [
            "You can contact support at support@example.com",
            "Our support team is available from 9 AM to 6 PM."
        ]
    },

    {
        "tag": "motivation",
        "patterns": [
            "Motivate me",
            "I feel tired",
            "Say something inspiring"
        ],
        "responses": [
            "Every expert was once a beginner.",
            "Keep learning — small progress is still progress.",
            "Consistency beats talent when talent doesn't work hard."
        ]
    },

    {
        "tag": "goodbye",
        "patterns": [
            "Bye",
            "Goodbye",
            "See you later"
        ],
        "responses": [
            "Happy to help! Have a great day.",
            "Goodbye! Reach out anytime."
        ]
    }
]

# -----------------------------------
# 2. PREPARE TRAINING DATA
# -----------------------------------

corpus = []
tags = []

for intent in intents:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

# -----------------------------------
# 3. TEXT PREPROCESSING
# -----------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join(
        [char for char in text if char not in string.punctuation]
    )

    return text

processed_corpus = [preprocess_text(doc) for doc in corpus]

# -----------------------------------
# 4. VECTORIZE TEXT
# -----------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_corpus)

# -----------------------------------
# 5. CHATBOT RESPONSE FUNCTION
# -----------------------------------

def get_bot_response(user_input):

    processed_input = preprocess_text(user_input)

    # Convert user input into vector
    user_vec = vectorizer.transform([processed_input])

    # Compare similarity
    similarities = cosine_similarity(user_vec, X)

    # Get best matching sentence
    best_match_index = similarities.argmax()

    highest_similarity = similarities[0][best_match_index]

    # Confidence threshold
    if highest_similarity < 0.3:
        return "I'm sorry, I didn't understand that. Could you rephrase?"

    # Get matching tag
    matched_tag = tags[best_match_index]

    # Return random response
    for intent in intents:
        if intent['tag'] == matched_tag:
            return random.choice(intent['responses'])

# -----------------------------------
# 6. CHATBOT INTERFACE
# -----------------------------------

print("=" * 50)
print(" CUSTOMER SERVICE CHATBOT ")
print("=" * 50)
print("Type 'quit' to exit.\n")

while True:

    user_message = input("You: ")

    if user_message.lower() == "quit":
        print("Bot: Goodbye!")
        break

    response = get_bot_response(user_message)

    print("Bot:", response)