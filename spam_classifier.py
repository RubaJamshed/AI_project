import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ========== STEP 1: LOAD THE DATA ==========
print("Step 1: Loading data...")
try:
    df = pd.read_csv('sms spam classifier/spam.csv', encoding='latin-1')
except:
    try:
        df = pd.read_csv('spam.csv')
    except:
        print("ERROR: Could not find dataset file!")
        print("Make sure spam.csv or SMSSpamCollection is in the same folder as this script")
        exit()

if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
elif 0 in df.columns and 1 in df.columns:
    df.columns = ['label', 'message']

print(f"Loaded {len(df)} messages")
print(f"Spam: {sum(df['label'] == 'spam')} messages")
print(f"Ham (not spam): {sum(df['label'] == 'ham')} messages")
print()

# ========== STEP 2: PREPARE THE DATA ==========
print("Step 2: Preparing data...")
# Convert labels to numbers (spam=1, ham=0)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# ========== STEP 3: CONVERT TEXT TO NUMBERS ==========
print("Step 3: Converting text to numbers...")
#this creates a "bag of words" - counts how many times each word appears
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])  # X = the messages as numbers
y = df['label_num']  # y = the labels (spam or not)

print(f"Found {len(vectorizer.get_feature_names_out())} unique words")
print()

# ========== STEP 4: SPLIT DATA FOR TRAINING AND TESTING ==========
print("Step 4: Splitting data...")
#80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]} messages")
print(f"Testing on {X_test.shape[0]} messages")
print()

# ========== STEP 5: TRAIN THE MODEL ==========
print("Step 5: Training the model...")
model = MultinomialNB()  # Naive Bayes algorithm
model.fit(X_train, y_train)
print("Training complete!")
print()

# ========== STEP 6: TEST THE MODEL ==========
print("Step 6: Testing the model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print()

# ========== STEP 7: SHOW RESULTS ==========
print("Step 7: Detailed results:")
print("-" * 40)
# Confusion matrix shows:
# [True Ham, False Spam]
# [False Ham, True Spam]
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print("            Predicted Ham  Predicted Spam")
print(f"Actual Ham      {cm[0][0]}            {cm[0][1]}")
print(f"Actual Spam     {cm[1][0]}            {cm[1][1]}")
print()

# Calculate precision and recall
tn, fp, fn, tp = cm.ravel()  # tn=true negative, fp=false positive, etc.
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision * 100:.2f}% (Of messages marked spam, how many were actually spam)")
print(f"Recall: {recall * 100:.2f}% (Of actual spam, how many did we catch)")
print(f"F1 Score: {f1 * 100:.2f}% (Balance between precision and recall)")
print()

# ========== STEP 8: TEST WITH CUSTOM MESSAGES ==========
print("=" * 50)
print("TEST YOUR OWN MESSAGES!")
print("=" * 50)

while True:
    print("\nEnter a message to check if it's spam (or 'quit' to exit):")
    user_message = input("> ")
    
    if user_message.lower() == 'quit':
        break
    
    #converts our message to numbers using the same vectorizer
    user_message_vector = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vector)[0]
    probability = model.predict_proba(user_message_vector)[0]
    
    if prediction == 1:
        print(f"⚠️  SPAM DETECTED! (Confidence: {probability[1]*100:.1f}%)")
    else:
        print(f"✓ NOT SPAM (Confidence: {probability[0]*100:.1f}%)")

