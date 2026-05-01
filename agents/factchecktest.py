from bs4 import BeautifulSoup
import json


def extract_structures(html):
    soup = BeautifulSoup(html, "lxml")

    tables = soup.find_all("table")
    infoboxes = soup.find_all("table", {"class": "infobox"})

    return tables, infoboxes


def table_to_facts(table):
    facts = []

    rows = table.find_all("tr")
    if not rows:
        return facts

    headers = [th.get_text(strip=True).lower() for th in rows[0].find_all("th")]
    if not headers:
        return facts

    for row in rows[1:]:
        cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cols) != len(headers):
            continue

        entity = cols[0].lower().replace(" ", "_")

        for h, v in zip(headers[1:], cols[1:]):
            if not v:
                continue

            facts.append({
                "entity": entity,
                "attribute": h,
                "value": v,
                "source": "table",
                "confidence": 0.95
            })

    return facts


def infobox_to_facts(infobox):
    facts = []

    rows = infobox.find_all("tr")
    entity = None

    for row in rows:
        header = row.find("th")
        value = row.find("td")

        if header and not value:
            entity = header.get_text(strip=True).lower().replace(" ", "_")

        elif header and value:
            val_text = value.get_text(strip=True)
            if not val_text:
                continue

            facts.append({
                "entity": entity,
                "attribute": header.get_text(strip=True).lower(),
                "value": val_text,
                "source": "infobox",
                "confidence": 0.98
            })

    return facts


def normalize_fact(f):
    return {
        "entity": f["entity"].lower().replace(" ", "_") if f["entity"] else "unknown",
        "attribute": f["attribute"].lower(),
        "value": str(f["value"]).strip(),
        "confidence": f.get("confidence", 0.8),
        "source": f.get("source", "unknown")
    }


def process_single_page(html):
    tables, infoboxes = extract_structures(html)

    print(f"\n=== STRUCTURE SUMMARY ===")
    print(f"Tables found: {len(tables)}")
    print(f"Infoboxes found: {len(infoboxes)}")

    all_facts = []

    for i, table in enumerate(tables[:3]):  # limit for readability
        facts = table_to_facts(table)
        print(f"\n--- Table {i+1}: {len(facts)} facts ---")
        all_facts.extend(facts[:10])  # preview only

    for i, infobox in enumerate(infoboxes[:1]):
        facts = infobox_to_facts(infobox)
        print(f"\n--- Infobox {i+1}: {len(facts)} facts ---")
        all_facts.extend(facts)

    # normalize
    all_facts = [normalize_fact(f) for f in all_facts]

    print("\n=== SAMPLE FACTS ===")
    for f in all_facts[:20]:
        print(json.dumps(f, indent=2))


# ---- USAGE ----
with open("anvil.html", "r", encoding="utf-8") as f:
    html = f.read()

process_single_page(html)