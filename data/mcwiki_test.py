import requests
import os
import re
import time

API_URL = "https://minecraft.wiki/api.php"

def get_all_pages():
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": "max",
        "apnamespace": 0,  # ONLY main content pages
        "apfilterredir": "nonredirects",  # exclude redirects
        "format": "json"
    }

    while True:
        res = requests.get(API_URL, params=params).json()
        pages.extend([p["title"] for p in res["query"]["allpages"]])

        if "continue" in res:
            params.update(res["continue"])
        else:
            break

    return pages


def get_page_text(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }

    res = requests.get(API_URL, params=params).json()
    pages = res["query"]["pages"]
    return next(iter(pages.values())).get("extract", "")


def safe_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def save_pages(pages):
    os.makedirs("corpus/wiki_pages", exist_ok=True)

    for title in pages:

        if ":" in title:
            continue

        text = get_page_text(title)

        filename = safe_filename(title) + ".txt"
        with open(os.path.join("corpus/wiki_pages", filename), "w", encoding="utf-8") as f:
            f.write(text)

        time.sleep(0.5)  # IMPORTANT: avoid rate limits


pages = get_all_pages()
save_pages(pages)