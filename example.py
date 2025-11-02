"""
Simple Python SDK to validate an OpenAI API key and make test requests.
"""

from openai import OpenAI, APIStatusError


class OpenAISDK:
    def __init__(self, api_key: str):
        """
        Initialize the SDK with a given OpenAI API key.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.client = OpenAI(api_key=api_key)

    def check_api_key(self) -> bool:
        """
        Check whether the provided API key is valid by making a lightweight API call.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            # Make a minimal test request — a small completion call
            self.client.models.list()
            return True
        except APIStatusError as e:
            # If the API key is invalid or unauthorized
            if e.status_code == 401:
                return False
            raise  # re-raise unexpected errors
        except Exception:
            return False


if __name__ == "__main__":
    import os

    # Example usage
    api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")

    sdk = OpenAISDK(api_key)
    if sdk.check_api_key():
        print("✅ API key is valid.")
    else:
        print("❌ Invalid or unauthorized API key.")
