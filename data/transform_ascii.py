import os
import unicodedata

FOLDER_PATH = r"C:\Users\qyptn\Revature_Training\minecraft\data\corpus\wiki_pages"


# ---------------------------
# ASCII CLEANER
# ---------------------------
def to_ascii(text: str) -> str:
    # normalize unicode (× → x, etc.)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


# ---------------------------
# CLEAN FILE CONTENT
# ---------------------------
def clean_file(filepath: str):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    cleaned = to_ascii(content)

    with open(filepath, "w", encoding="ascii", errors="ignore") as f:
        f.write(cleaned)


# ---------------------------
# CLEAN FILE NAME (TITLE)
# ---------------------------
def clean_filename(filepath: str) -> str:
    dir_name = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)

    name, ext = os.path.splitext(file_name)

    clean_name = to_ascii(name)
    clean_name = clean_name.replace(" ", "_")

    new_filename = clean_name + ext
    new_path = os.path.join(dir_name, new_filename)

    if new_path == filepath:
        return filepath

    if os.path.exists(new_path):
        base = clean_name
        i = 1
        while True:
            candidate = os.path.join(dir_name, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                new_path = candidate
                break
            i += 1

    os.rename(filepath, new_path)
    print(f"Renamed: {file_name} → {os.path.basename(new_path)}")

    return new_path


# ---------------------------
# CLEAN FOLDER
# ---------------------------
def clean_folder(folder: str):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)

                print(f"Processing: {path}")

                # 1. clean filename first
                path = clean_filename(path)

                # 2. clean file content
                clean_file(path)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    clean_folder(FOLDER_PATH)
    print("Done cleaning ASCII characters (files + titles).")