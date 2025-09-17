import requests
import os
import time
from dotenv import load_dotenv

# --- The Correction is in this section ---
# We'll define the project root path more robustly.
# This code assumes the script is in 'project_root/py_classes/utils/'
# and the .env is in 'project_root/'
try:
    # Go up two directories from the current script's location to find the project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')
except NameError:
    # This handles cases where the script is run in an interactive environment
    # where __file__ might not be defined.
    PROJECT_ROOT = os.getcwd()
    ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')
# --- End of Correction ---

class BraveSearchAPI:
    """
    A static class to interact with the Brave Search API.
    """
    BASE_URL = "https://api.search.brave.com/res/v1"

    @staticmethod
    def search(api_key: str, endpoint: str, params: dict):
        """
        A single method to call any Brave Search API endpoint.
        """
        if not api_key or "YOUR_API_KEY" in api_key:
            print("Error: API key is missing or is a placeholder. Please provide a valid API key.")
            return None

        url = f"{BraveSearchAPI.BASE_URL}/{endpoint}/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }

        try:
            print(f"Calling endpoint: {endpoint} with query: '{params.get('q')}'")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
        
        return None

if __name__ == '__main__':
    print("--- Testing API Calls ---")

    # Check if the .env file exists at the calculated path
    if not os.path.exists(ENV_FILE_PATH):
        print(f"Error: .env file not found at the expected path: {ENV_FILE_PATH}")
        print("Please ensure your .env file is in the project root directory.")
    else:
        # Load environment variables from the .env file
        load_dotenv(ENV_FILE_PATH)
        API_KEY = os.getenv("BRAVE_API_KEY")

        if not API_KEY:
            print("NOTE: BRAVE_API_KEY variable not found inside the .env file. Aborting tests.")
        else:
            print("NOTE: A 1-second delay is added between calls to respect the free plan's rate limit.")

            # --- Test Cases ---
            # ... (The rest of your test cases remain exactly the same) ...

            # 1. Web Search
            print("\n--- 1. Testing Web Search ---")
            web_params = {"q": "python requests library"}
            web_results = BraveSearchAPI.search(API_KEY, "web", web_params)
            if web_results:
                print("Web Search Successful. Result keys:", web_results.keys())
            
            time.sleep(1)

            # 2. Image Search
            print("\n--- 2. Testing Image Search ---")
            image_params = {"q": "mountain landscape"}
            image_results = BraveSearchAPI.search(API_KEY, "images", image_params)
            if image_results:
                print("Image Search Successful. Result keys:", image_results.keys())

            time.sleep(1)

            # 3. News Search
            print("\n--- 3. Testing News Search ---")
            news_params = {"q": "latest advancements in AI"}
            news_results = BraveSearchAPI.search(API_KEY, "news", news_params)
            if news_results:
                print("News Search Successful. Result keys:", news_results.keys())

            time.sleep(1)

            # 4. Video Search
            print("\n--- 4. Testing Video Search ---")
            video_params = {"q": "how to make sourdough bread"}
            video_results = BraveSearchAPI.search(API_KEY, "videos", video_params)
            if video_results:
                print("Video Search Successful. Result keys:", video_results.keys())
            
            time.sleep(1)
                
            # 5. Suggestions (EXPECTED TO FAIL ON FREE PLAN)
            print("\n--- 5. Testing Suggestions ---")
            print("(This test is expected to fail with a 400 error on the Free AI plan)")
            suggest_params = {"q": "how to learn pyth"}
            suggest_results = BraveSearchAPI.search(API_KEY, "suggest", suggest_params)
            if suggest_results:
                print("Suggestion Search Successful. Results:", suggest_results)

            time.sleep(1)

            # 6. Spellcheck
            print("\n--- 6. Testing Spellcheck ---")
            spellcheck_params = {"q": "pyhton languaeg"}
            spellcheck_results = BraveSearchAPI.search(API_KEY, "spellcheck", spellcheck_params)
            if spellcheck_results:
                print("Spellcheck Successful. Results:", spellcheck_results)