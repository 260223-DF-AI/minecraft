import requests
from bs4 import BeautifulSoup

url = "https://learn.microsoft.com/en-us/minecraft/creator/documents/redstoneguide?view=minecraft-bedrock-stable"

def url_to_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # remove junk elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # extract text
    text = soup.get_text(separator="\n")

    # clean whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join([line for line in lines if line])

    return clean_text


def save_to_txt(url: str, output_file="redstoneguide.txt"):
    text = url_to_text(url)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved → {output_file}")


save_to_txt(url)