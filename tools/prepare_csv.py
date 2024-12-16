import re
import csv

def parse_wa_file(wa_file_path, output_csv_path):
    with open(wa_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentence_pattern = re.compile(
        r'<sentence id="(\d+)" status="">\s*// (.*?)\s*// (.*?)\s*<source>(.*?)</source>\s*<translation>(.*?)</translation>\s*<alignment>(.*?)</alignment>',
        re.DOTALL)
    chunk_pattern = re.compile(r'(\d+) ([^:]+) :')
    alignment_pattern = re.compile(r'([\d ]+)<==>([\d ]+) // (\w+) // (\w+) // (.*?) <==> (.*?)')
    data_rows = []

    for match in sentence_pattern.finditer(content):
        sentence_id = match.group(1)
        sentence1 = match.group(2).strip()
        sentence2 = match.group(3).strip()
        source_data = match.group(4)
        translation_data = match.group(5)
        alignment_data = match.group(6)

        source_chunks = {int(m.group(1)): m.group(2).strip() for m in chunk_pattern.finditer(source_data)}
        translation_chunks = {int(m.group(1)): m.group(2).strip() for m in chunk_pattern.finditer(translation_data)}

        for alignment in alignment_pattern.finditer(alignment_data):
            source_indices = alignment.group(1).split()
            translation_indices = alignment.group(2).split()
            y_type = alignment.group(3)
            y_score = alignment.group(4)

            source_text = " ".join([source_chunks.get(int(idx), "") for idx in source_indices])
            translation_text = " ".join(
                [translation_chunks.get(int(idx), "") for idx in translation_indices])

            # Add data row
            data_rows.append({
                'x1': source_text,
                'x2': translation_text,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'y_type': y_type,
                'y_score': y_score
            })

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['x1', 'x2', 'sentence1', 'sentence2', 'y_type', 'y_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_rows)
        print('CSV saved')


wa_file_path = '.wa'
output_csv_path = '.csv'
parse_wa_file(wa_file_path, output_csv_path)