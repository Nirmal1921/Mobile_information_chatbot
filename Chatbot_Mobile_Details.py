import requests
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from transformers import pipeline
import numpy as np

# GSMArena Scraping URL
GSMARENA_URL = "https://www.gsmarena.com/"
CACHE_EXPIRY = timedelta(hours=1)
cache = {}
cache_time = None

# NLP Model for Query Understanding
nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intent Categories
INTENT_LABELS = ["latest phones", "phone details", "compare phones", "trending phones", "greeting", "help"]

"""def fetch_latest_phones() -> List[Dict[str, Any]]:
    Fetches the latest phone models from GSMArena.
    global cache, cache_time
    
    if cache and cache_time and datetime.now() - cache_time < CACHE_EXPIRY:
        print("Returning cached phone data")
        return cache
    
    try:
        response = requests.get(GSMARENA_URL + "search.php3?sSort=newest", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        phones = []
        phone_items = soup.select('.makers ul li')

        for item in phone_items[:10]:  # Increased to 10 phones
            link_elem = item.select_one('a')
            name_elem = item.select_one('a strong')

            if not link_elem or not name_elem:
                continue

            name = name_elem.text.strip()
            link = GSMARENA_URL + link_elem['href']

            phones.append({"name": name, "link": link})

        cache = phones
        cache_time = datetime.now()
        return phones
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest phones: {e}")
        return []"""

def fetch_latest_phones() -> List[Dict[str, Any]]:
    """Fetches the top 10 trending phones by daily interest from GSMArena."""
    try:
        response = requests.get(GSMARENA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the section with top trending phones
        trending_section = soup.select_one("div.module.module-rankings.s3 table.module-fit.green tbody")

        if not trending_section:
            return []

        # Extract top 10 phone details
        trending_phones = []
        seen_links = set()

        for row in trending_section.select("tr"):
            name_elem = row.select_one("th[headers='th3b'] a")
            hits_elem = row.select_one("td[headers='th3c']")
            if name_elem and hits_elem:
                name = name_elem.text.strip()
                hits = hits_elem.text.strip()
                link = GSMARENA_URL + name_elem['href']
                if link not in seen_links:
                    seen_links.add(link)
                    trending_phones.append({"name": name, "hits": hits, "link": link})
                if len(trending_phones) == 10:
                    break

        return trending_phones
    except requests.exceptions.RequestException as e:
        print(f"Error fetching trending phones: {e}")
        return []

def fetch_phone_details(phone_name: str) -> str:
    """Fetches detailed phone specifications from GSMArena."""
    search_url = GSMARENA_URL + f"results.php3?sQuickSearch=yes&sName={phone_name.replace(' ', '+')}"
    try:
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        first_result = soup.select_one('.makers ul li a')
        if not first_result:
            return f"I couldn't find details for {phone_name}. Please check the model name."

        phone_link = GSMARENA_URL + first_result['href']
        response = requests.get(phone_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract phone specifications using correct table structure
        specs = {}
        spec_sections = soup.select('table tr')

        for row in spec_sections:
            key_elem = row.select_one('td.ttl')
            value_elem = row.select_one('td.nfo')
            if key_elem and value_elem:
                key = key_elem.text.strip()
                value = value_elem.text.strip()
                if key not in specs:
                    specs[key] = value  # Store only the first occurrence to avoid duplicates

        # Format extracted details
        details = f"📱 {phone_name} Details:\n\n"
        
        # Group specifications into categories for better readability
        categories = {
            "Display": ["display", "Display", "size", "Size", "resolution", "Resolution", "type", "Type"],
            "Platform": ["platform", "Platform", "OS", "chipset", "Chipset", "CPU", "GPU"],
            "Camera": ["camera", "Camera", "main camera", "Main Camera", "selfie", "Selfie"],
            "Battery": ["battery", "Battery", "charging", "Charging"],
            "Memory": ["memory", "Memory", "RAM", "storage", "Storage", "card slot", "Card slot"]
        }
        
        # Add specifications by category
        for category, keys in categories.items():
            category_specs = []
            for spec_key in specs:
                if any(key in spec_key for key in keys):
                    category_specs.append(f"{spec_key}: {specs[spec_key]}")
            
            if category_specs:
                details += f"🔹 {category}:\n"
                details += "\n".join([f"   - {spec}" for spec in category_specs])
                details += "\n\n"
        
        # Add any remaining specifications not in categories
        other_specs = []
        for spec_key in specs:
            if not any(key in spec_key for category_keys in categories.values() for key in category_keys):
                other_specs.append(f"{spec_key}: {specs[spec_key]}")
        
        if other_specs:
            details += "🔹 Other Specifications:\n"
            details += "\n".join([f"   - {spec}" for spec in other_specs])
            details += "\n\n"

        details += f"🔗 More info: {phone_link}"
        return details
    except requests.exceptions.RequestException as e:
        return f"Error fetching phone details: {e}"

def compare_phones(phone1: str, phone2: str) -> str:
    """Compares specifications of two phones from GSMArena."""
    def fetch_phone_specs(phone_name):
        """Helper function to fetch phone details with improved search."""
        # Clean up phone name for better search results
        search_term = phone_name.strip()
        
        # Handle special cases for iPhone model variants
        if "iphone" in search_term.lower():
            # Check if this is a Pro/Pro Max/Plus variant
            model_variants = ["pro max", "pro", "plus", "mini"]
            base_model = None
            variant = None
            
            for v in model_variants:
                if v in search_term.lower():
                    variant = v
                    base_model = search_term.lower().split(v)[0].strip()
                    break
                    
            if variant:
                # Make sure the variant is included in the search term
                search_term = f"{base_model} {variant}"
                
        search_url = GSMARENA_URL + f"results.php3?sQuickSearch=yes&sName={search_term.replace(' ', '+')}"

        try:
            response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get all search results to better match the exact phone model
            results = soup.select('.makers ul li a')
            if not results:
                return None, None, None
                
            # Find the best match for the phone name
            best_match = None
            best_match_link = None
            actual_name = None
            
            # First try to find an exact match for the full phone name
            for result in results:
                result_name = result.select_one('strong').text.strip().lower()
                search_name = search_term.lower()
                
                # Special handling for iPhone Pro/Pro Max models
                if "iphone" in search_name:
                    # For iPhone, match exact model number and variant
                    iphone_model = re.search(r'iphone\s+(\d+)', search_name)
                    if iphone_model:
                        model_num = iphone_model.group(1)
                        
                        # Match exact variant if specified
                        if "pro max" in search_name and f"iphone {model_num} pro max" in result_name:
                            best_match = result
                            actual_name = result.select_one('strong').text.strip()
                            break
                        elif "pro" in search_name and "pro max" not in search_name and f"iphone {model_num} pro" in result_name:
                            best_match = result
                            actual_name = result.select_one('strong').text.strip()
                            break
                        elif "plus" in search_name and f"iphone {model_num} plus" in result_name:
                            best_match = result
                            actual_name = result.select_one('strong').text.strip()
                            break
                        elif "mini" in search_name and f"iphone {model_num} mini" in result_name:
                            best_match = result
                            actual_name = result.select_one('strong').text.strip()
                            break
                        # Base model (no variant)
                        elif all(v not in search_name for v in ["pro", "plus", "mini"]) and all(v not in result_name for v in ["pro", "plus", "mini"]):
                            if f"iphone {model_num}" in result_name:
                                best_match = result
                                actual_name = result.select_one('strong').text.strip()
                                break
                else:
                    # For other phones, look for the best match
                    if search_name in result_name or result_name in search_name:
                        best_match = result
                        actual_name = result.select_one('strong').text.strip()
                        break
            
            # If no exact match found, use the first result
            if not best_match and results:
                best_match = results[0]
                actual_name = best_match.select_one('strong').text.strip()
            
            if not best_match:
                return None, None, None
                
            phone_link = GSMARENA_URL + best_match['href']
            
            # Fetch the phone details
            response = requests.get(phone_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract phone specifications
            specs = {}
            spec_sections = soup.select('table tr')

            for row in spec_sections:
                key_elem = row.select_one('td.ttl')
                value_elem = row.select_one('td.nfo')
                if key_elem and value_elem:
                    key = key_elem.text.strip().lower()
                    value = value_elem.text.strip()
                    if key not in specs:
                        specs[key] = value

            return specs, phone_link, actual_name

        except requests.exceptions.RequestException as e:
            return None, None, None

    specs1, link1, actual_name1 = fetch_phone_specs(phone1)
    specs2, link2, actual_name2 = fetch_phone_specs(phone2)

    if not specs1 or not specs2:
        return f"I couldn't fetch details for one or both phones: {phone1} and {phone2}. Please check the model names."
    
    # Use actual names from GSMArena if available
    display_name1 = actual_name1 if actual_name1 else phone1
    display_name2 = actual_name2 if actual_name2 else phone2
    
    # Ensure we're not comparing the same phone to itself
    if link1 == link2:
        return f"It looks like both '{phone1}' and '{phone2}' refer to the same phone model: {display_name1}. Please specify different models to compare."
        
    # Define comparison categories with multiple possible keys for each section
    comparison_sections = {
        "Display": ["display", "size", "resolution", "type"],
        "Platform": ["platform", "chipset", "cpu", "gpu", "os"],
        "Main Camera": ["main camera", "camera", "triple", "quad", "dual"],
        "Battery": ["battery", "charging"],
        "Memory": ["memory", "ram", "internal"],
        "OS": ["os"],
        "Chipset": ["chipset"]
    }

    # Compare specifications
    comparison = f"📱 {display_name1} vs {display_name2} Comparison:\n\n"

    for section, keys in comparison_sections.items():
        # Get values for each phone, joining multiple matches
        vals1 = [specs1.get(k, "N/A") for k in keys if k in specs1]
        vals2 = [specs2.get(k, "N/A") for k in keys if k in specs2]
        
        val1 = " | ".join(vals1) if vals1 else "N/A"
        val2 = " | ".join(vals2) if vals2 else "N/A"

        # Fix Camera section
        if section == "Main Camera":
            camera_keys = ["main camera", "camera", "triple", "quad", "dual"]
            raw_val1 = " | ".join([specs1.get(k, "N/A") for k in camera_keys if k in specs1])
            raw_val2 = " | ".join([specs2.get(k, "N/A") for k in camera_keys if k in specs2])

            # Use regex to remove "Photo / Video" if present
            val1 = re.sub(r"Photo / Video\s*\|\s*", "", raw_val1).strip()
            val2 = re.sub(r"Photo / Video\s*\|\s*", "", raw_val2).strip()

        # Fix Battery section
        if section == "Battery":
            battery_keys = ["battery", "charging"]
            val1 = " | ".join([specs1.get(k, "N/A") for k in battery_keys if k in specs1])
            val2 = " | ".join([specs2.get(k, "N/A") for k in battery_keys if k in specs2])

        # Only add sections that have at least some information
        if not (val1 == "N/A" and val2 == "N/A"):
            comparison += f"🔹 {section}:\n   - {display_name1}: {val1}\n   - {display_name2}: {val2}\n\n"

    comparison += f"🔗 More info:\n   - {display_name1}: {link1}\n   - {display_name2}: {link2}"
    return comparison

def extract_phone_names_for_comparison(query: str) -> Tuple[str, str]:
    """Extracts two phone names for comparison from the user query with improved handling."""
    # Clean the query
    query = query.lower().strip()
    
    # Pattern for full phone names with brand and model
    phone_patterns = [
        # "compare X and Y"
        r"compare\s+((?:iphone|samsung|xiaomi|redmi|huawei|oppo|vivo|oneplus|google|pixel|realme|motorola|sony|nokia|lg)[\w\s\d\+]+?)\s+(?:and|vs\.?|with|to)\s+((?:iphone|samsung|xiaomi|redmi|huawei|oppo|vivo|oneplus|google|pixel|realme|motorola|sony|nokia|lg)[\w\s\d\+]+?)(?:\s|$|\.)",
        # "X vs Y"
        r"((?:iphone|samsung|xiaomi|redmi|huawei|oppo|vivo|oneplus|google|pixel|realme|motorola|sony|nokia|lg)[\w\s\d\+]+?)\s+(?:vs\.?|versus)\s+((?:iphone|samsung|xiaomi|redmi|huawei|oppo|vivo|oneplus|google|pixel|realme|motorola|sony|nokia|lg)[\w\s\d\+]+?)(?:\s|$|\.)"
    ]
    
    # Try each pattern
    for pattern in phone_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            phone1 = match.group(1).strip()
            phone2 = match.group(2).strip()
            
            # Special handling for iPhone variants
            if "iphone" in phone1.lower():
                phone1 = polish_iphone_name(phone1)
            if "iphone" in phone2.lower():
                phone2 = polish_iphone_name(phone2)
                
            return phone1, phone2
    
    # Fallback approach - split by keywords
    for separator in [" and ", " vs ", " versus ", " with ", " to "]:
        if separator in query:
            parts = query.split(separator, 1)
            if len(parts) == 2:
                # Process each part to extract phone names
                brands = ["iphone", "samsung", "xiaomi", "redmi", "huawei", "oppo", "vivo", "oneplus", "google", "pixel", "realme", "motorola", "sony", "nokia", "lg"]
                
                # For each brand, look for its full mention in each part
                phone1, phone2 = "", ""
                
                for brand in brands:
                    if brand in parts[0].lower():
                        # Extract brand and the text following it
                        start_idx = parts[0].lower().find(brand)
                        phone1 = parts[0][start_idx:].strip()
                        # If iPhone, make sure to get the full model
                        if brand == "iphone":
                            phone1 = polish_iphone_name(phone1)
                        break
                        
                for brand in brands:
                    if brand in parts[1].lower():
                        # Extract brand and the text following it
                        start_idx = parts[1].lower().find(brand)
                        phone2 = parts[1][start_idx:].strip()
                        # If iPhone, make sure to get the full model
                        if brand == "iphone":
                            phone2 = polish_iphone_name(phone2)
                        break
                        
                if phone1 and phone2:
                    return phone1, phone2
    
    # If all else fails
    return "", ""

def polish_iphone_name(phone_name: str) -> str:
    """Standardizes iPhone model names to improve matching accuracy."""
    phone_name = phone_name.lower().strip()
    
    # Match iPhone model number
    model_match = re.search(r'iphone\s*(\d+)', phone_name)
    if not model_match:
        return phone_name
        
    model_num = model_match.group(1)
    
    # Check for variants
    if "pro max" in phone_name:
        return f"iPhone {model_num} Pro Max"
    elif "pro" in phone_name:
        return f"iPhone {model_num} Pro"
    elif "plus" in phone_name:
        return f"iPhone {model_num} Plus"
    elif "mini" in phone_name:
        return f"iPhone {model_num} Mini"
    else:hi
        return f"iPhone {model_num}"
    
def extract_phone_name(query: str) -> str:
    """Extracts a phone name from the user query."""
    # Pattern to match "about X" or "tell me about X"
    about_pattern = r"(?:about|tell me about|info on|details on|details for|details about)\s+([^,\.]+)(?:\s|$|\.)"
    
    # Check for about pattern
    about_match = re.search(about_pattern, query, re.IGNORECASE)
    if about_match:
        return about_match.group(1).strip()
    
    # Check for common phone brands followed by text
    brands = ["samsung", "apple", "iphone", "xiaomi", "huawei", "oppo", "vivo", "oneplus", "google", "pixel", "realme", "motorola", "sony", "nokia", "lg"]
    for brand in brands:
        brand_pattern = fr'{brand}\s+[\w\d\s]+?(?=\s|$|\.)'
        brand_match = re.search(brand_pattern, query, re.IGNORECASE)
        if brand_match:
            return brand_match.group(0).strip()
    
    # If no pattern match, look for anything that might be a phone name
    # This is a fallback and might not be accurate
    words = query.split()
    for i in range(len(words) - 1):
        segment = ' '.join(words[i:i+3])  # Try up to 3 consecutive words
        if any(brand in segment.lower() for brand in brands):
            return segment
    
    return ""

def detect_intent(user_input: str) -> Dict[str, Any]:
    """Detects the user's intent using a transformer-based model and rule-based patterns."""
    # First check for common patterns using regex
    if re.search(r'\b(latest|newest|new|recent)\s+phones?\b', user_input, re.IGNORECASE):
        return {"intent": "latest phones", "confidence": 0.9, "query": user_input}
    
    if re.search(r'\b(trending|popular|top|hot)\s+phones?\b', user_input, re.IGNORECASE):
        return {"intent": "trending phones", "confidence": 0.9, "query": user_input}
    
    if re.search(r'\b(compare|vs\.?|versus)\b', user_input, re.IGNORECASE):
        phone1, phone2 = extract_phone_names_for_comparison(user_input)
        if phone1 and phone2:
            return {"intent": "compare phones", "confidence": 0.9, "query": user_input, "phone1": phone1, "phone2": phone2}
    
    # Check for phone details
    if re.search(r'\b(details|about|info|specifications|specs|features)\b', user_input, re.IGNORECASE):
        phone_name = extract_phone_name(user_input)
        if phone_name:
            return {"intent": "phone details", "confidence": 0.9, "query": user_input, "phone_name": phone_name}
    
    # Check for greetings
    if re.search(r'\b(hi|hello|hey|greetings)\b', user_input, re.IGNORECASE):
        return {"intent": "greeting", "confidence": 0.9, "query": user_input}
    
    # Check for help
    if re.search(r'\b(help|assistance|what can you do|how to use|features)\b', user_input, re.IGNORECASE):
        return {"intent": "help", "confidence": 0.9, "query": user_input}
    
    # Fallback to transformer model for complex queries
    result = nlp(user_input, INTENT_LABELS)
    
    logits = np.array(result["scores"])
    best_match_idx = np.argmax(logits)
    best_match = result['labels'][best_match_idx]
    confidence = logits[best_match_idx]
    
    # For certain intents, extract additional info if not already done
    if best_match == "phone details":
        phone_name = extract_phone_name(user_input)
        return {"intent": best_match, "confidence": confidence, "query": user_input, "phone_name": phone_name}
    elif best_match == "compare phones":
        phone1, phone2 = extract_phone_names_for_comparison(user_input)
        return {"intent": best_match, "confidence": confidence, "query": user_input, "phone1": phone1, "phone2": phone2}
    
    return {"intent": best_match, "confidence": confidence, "query": user_input}

def chatbot_response(user_input: str) -> str:
    """Processes user input and generates a response."""
    intent_data = detect_intent(user_input)
    intent = intent_data["intent"]
    confidence = intent_data["confidence"]
    query = intent_data["query"]
    
    if confidence < 0.4:
        return "I'm not sure I understand. Can you provide more details? You can ask about latest phones, trending phones, a specific model, or comparisons."
    
    if intent == 'latest phones':
        phones = fetch_latest_phones()
        if not phones:
            return "I couldn't fetch the latest phone models at the moment. Please try again later."
        response = "Here are the latest phone models:\n\n"
        response += '\n'.join([f"📱 {p['name']} - More info: {p['link']}" for p in phones])
        return response
    
    elif intent == 'trending phones':
        phones = fetch_latest_phones()
        if not phones:
            return "I couldn't fetch the trending phone models at the moment. Please try again later."
        response = "Here are the top 10 trending phones by daily interest:\n\n"
        response += '\n'.join([f"📱 {idx}. {p['name']} ({p['hits']} hits) - More info: {p['link']}" for idx, p in enumerate(phones, 1)])
        return response
    
    elif intent == "phone details":
        phone_name = intent_data.get("phone_name", "")
        if not phone_name:
            phone_name = extract_phone_name(query)
        
        if not phone_name:
            return "I need a specific phone model to look up. Could you please specify the phone name?"
        
        return fetch_phone_details(phone_name)
    
    elif intent == "compare phones":
        phone1 = intent_data.get("phone1", "")
        phone2 = intent_data.get("phone2", "")
        
        if not phone1 or not phone2:
            phone1, phone2 = extract_phone_names_for_comparison(query)
        
        if not phone1 or not phone2:
            return "I need two specific phone models to compare. Could you please specify both phone names?"
        
        return compare_phones(phone1, phone2)
    
    elif intent == "greeting":
        return "Hello! I can help you with the latest phone models, trending phones, specifications, and comparisons. What would you like to know?"
    
    elif intent == "help":
        return """
I can help you with information about mobile phones. Here's what you can ask me:

1. Latest phones: "What are the latest phones?"
2. Trending phones: "What are the trending phones right now?"
3. Phone details: "Tell me about iPhone 15" or "Samsung Galaxy S24 specs"
4. Compare phones: "Compare iPhone 15 and Samsung Galaxy S24" or "iPhone 15 vs Galaxy S24"

Just ask your question and I'll do my best to help!
"""
    
    return "I'm not sure I understand. Try asking about the latest phone models, trending phones, a specific model, or phone comparisons!"

# Running the chatbot
if __name__ == "__main__":
    print("Phone Information Chatbot")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 50)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Thank you for chatting! Goodbye!")
            break
        print(f"Bot: {chatbot_response(user_input)}")
