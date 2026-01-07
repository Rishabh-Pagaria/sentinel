from models.encoders import DebertaPhishingEncoder

EMAIL_TEXT = """
Dear user,
Your account has been suspended. Click the link below to verify.
"""

encoder = DebertaPhishingEncoder()
encoder.load()

p = encoder.predict_proba_email(EMAIL_TEXT)

print(f"Phishing probability: {p:.4f}")
print("Prediction:", "PHISHING" if p >= 0.5 else "BENIGN")
