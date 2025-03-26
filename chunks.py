from pathlib import Path

def get_chunks(path):
    # Load all .txt files from mock_notes
    notes_dir = Path(path)
    chunks = []

    for file_path in notes_dir.glob("*.txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue  # skip malformed files
            project_name = lines[0].strip().replace("Account: ", "")
            contact_name = lines[1].strip().replace("Client Contact: ", "")
            note_body = "".join(lines[2:]).strip()
            chunks.append({
                "file": file_path.name,
                "project": project_name,
                "contact": contact_name,
                "text": note_body,
                "preview": "\n".join(note_body.splitlines()[:3])
            })
    
    return chunks

if __name__ == "__main__":
    import pandas as pd
    chunks = get_chunks('./mocks')
    chunks_df = pd.DataFrame(chunks)
    print(chunks_df.head())