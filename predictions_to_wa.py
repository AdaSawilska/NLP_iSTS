import re
import pandas as pd


def parse_gt_wa(gt_file):
    """Parses ground truth file for IDs and sentences."""
    gt_sentences = []
    with open(gt_file, 'r') as f:
        content = f.read()
        for sentence_match in re.finditer(r'<sentence id="(\d+)".*?</sentence>', content, re.DOTALL):
            sentence_content = sentence_match.group(0)
            source_match = re.search(r'// (.+)\n// (.+)\n', sentence_content)
            if source_match:
                gt_sentences.append({
                    'sentence_id': int(sentence_match.group(1)),
                    'source': source_match.group(1),
                    'translation': source_match.group(2),
                    'content': sentence_content
                })
    return gt_sentences


def get_sentence_with_id(gt_sentences, source, translation):
    """Gets matching sentence data from GT."""
    for gt in gt_sentences:
        if gt['source'] == source and gt['translation'] == translation:
            return gt
    return None


def parse_tokens(content, section_name):
    """Parse token mappings from section."""
    pattern = f'<{section_name}>\n(.*?)\n</{section_name}>'
    section_match = re.search(pattern, content, re.DOTALL)
    if not section_match:
        return {}
    tokens = {}
    for line in section_match.group(1).split('\n'):
        if line.strip():
            idx, rest = line.split(' ', 1)
            token = rest.split(' : ')[0]
            tokens[int(idx)] = token
    return tokens


def get_indices_from_words(word_string, token_map):
    """Get indices for words in the order they appear."""

    words = str(word_string).split()
    current_position = 1
    indices = []

    for word in words:
        while current_position <= max(token_map.keys()):
            if token_map.get(current_position) == word:
                indices.append(str(current_position))
                current_position += 1
                break
            current_position += 1

    if len(indices) == 0:
        indices = "0"
    return " ".join(indices)


def create_wa_file_from_predictions(df, gt_file, output_file):
    type_mapping = {0: 'EQUI', 1: 'OPPO', 2: 'SPE1', 3: 'SPE2',
                    4: 'SIMI', 5: 'REL', 6: 'NOALI'}
    df['alignment_type'] = df['predicted_type'].map(type_mapping)
    gt_sentences = parse_gt_wa(gt_file)

    with open(output_file, 'w') as f:
        for (source, translation), group in df.groupby(['sentence1', 'sentence2']):
            gt_data = get_sentence_with_id(gt_sentences, source, translation)
            if not gt_data:
                print(f"Warning: No matching sentence found for: {source} <==> {translation}")
                continue

            source_tokens = parse_tokens(gt_data['content'], 'source')
            translation_tokens = parse_tokens(gt_data['content'], 'translation')

            f.write(gt_data['content'].split('<alignment>')[0])
            f.write('<alignment>\n')

            for _, row in group.iterrows():
                src_words = str(row['x1'])
                tgt_words = str(row['x2'])

                src_indices = get_indices_from_words(src_words, source_tokens)
                tgt_indices = get_indices_from_words(tgt_words, translation_tokens)

                score = "NIL" if row.alignment_type in ["NOALI"] else str(row['predicted_score'])
                f.write(
                    f'{src_indices} <==> {tgt_indices} // {row["alignment_type"]} // {score} // {src_words} <==> {tgt_words}\n')

            f.write('</alignment>\n</sentence>\n\n')


if __name__ == "__main__":
    df = pd.read_csv('results/predictions_test_images.csv')
    create_wa_file_from_predictions(df,
                                    'data/Semeval2016/test/test_goldStandard/STSint.testinput.images.wa',
                                    'results/predictions_test_images.wa')