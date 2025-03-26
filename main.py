from chunks import get_chunks
from embeddings import compute_embeddings
from llm import generate_answer
from named_entity_utils import extract_account_names, extract_people_names
from retriever import retrieve

test_query = 'Did client contacts for Nova Expansion ever request access to staging environment?'

def run():
    people_names = extract_people_names(test_query)
    print(f"Names found: {', '.join(people_names)}")
    chunks = get_chunks('./mocks')
    print('chunks created')
    accounts = extract_account_names(test_query, chunks)
    metadata_matching_chunks = [chunk for chunk in chunks if chunk["contact"] in people_names or chunk["project"] in accounts] or chunks
    chunks_with_embeddings = compute_embeddings(metadata_matching_chunks, use_precomputed=False)
    print('embeddings calculated')
    matching_chunks = retrieve(test_query, chunks_with_embeddings)
    print('matching chunks found. Context:')
    context = "\n\n".join([f'project/account:{chunk["project"]}, client contact: {chunk["contact"]}, NOTES: {chunk["text"]}' for chunk in matching_chunks])
    print(context)

    print('\n\nANSWER:')
    answer = generate_answer(context, test_query)

    print(answer)
    
if __name__ == "__main__":
    run()