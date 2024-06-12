import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from extract_information2 import read_reports_from_folder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx

def extractive_summary(text, model, tokenizer):
    paragraphs = text.split('\n\n')
    summaries = []

    # Ignore the last paragraph if it's empty
    if not paragraphs[-1].strip():
        paragraphs = paragraphs[:-1]

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        inputs = tokenizer.batch_encode_plus(sentences, return_tensors='tf', max_length=512, truncation=True, padding='longest')
        outputs = model(inputs['input_ids'])
        sentence_embeddings = tf.reduce_mean(outputs[0], axis=1).numpy()
        similarity_matrix = cosine_similarity(sentence_embeddings)
        np.fill_diagonal(similarity_matrix, 0)

        # Use TextRank on the similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Rank sentences based on scores
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # Calculate the number of sentences to return
        num_sentences = int(len(sentences) * 0.5)

        # Add the summary of this paragraph to the list of summaries
        summaries.append(' '.join([s for (_, s) in ranked_sentences[:num_sentences]]))

    # Combine the summaries of all paragraphs into a single string
    return '\n\n'.join(summaries)

def main():
    folder_path = "E:/MSCS/NLP/Project/code/data/sample/input/"  # Update this with the path to your folder containing text files

    print("Starting to extract information from files...")

    all_reports = read_reports_from_folder(folder_path)

    print(f"Finished extracting information from {len(all_reports)} files.")

    # Load BioBERT
    model = TFBertModel.from_pretrained("monologg/biobert_v1.1_pubmed")
    tokenizer = BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

    # Generate summary for each report and save it to a text file
    for i, report in enumerate(all_reports):
        summary = extractive_summary(report, model, tokenizer)
        with open(f'E:/MSCS/NLP/Project/code/data/sample/output/summary_{i}.txt', 'w') as f:
            f.write(summary)

if __name__ == "__main__":
    main()