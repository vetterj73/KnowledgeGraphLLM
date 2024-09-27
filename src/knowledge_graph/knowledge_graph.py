"""Module from testing knowlege graphs and embedding with call data"""

import os
import base64
import pickle
import pandas as pd
import networkx as nx
import faiss
import numpy as np
import gzip
import tiktoken
from openai import OpenAI

# Paths to saved files
INDEX_PATH = 'call_log_index.bin'
MAPPING_PATH = 'call_log_id_to_entity.pkl'  # or 'id_to_entity.json'
GRAPH_PATH = 'call_log_graph.gpickle'
FILE_PATH = 'D:\\Work\\Mercury\\MercSIS2024.csv'
FILE_ENCODING = 'ISO-8859-1'
OPEN_API_KEY = os.environ['OPEN_API_KEY']  # noqa: E501
BATCH_SIZE_LIMIT = 500  # Maximum number of inputs per batch (adjust as needed)


def obj_to_txt(obj):
    """Convert an object to it's string representation."""
    message_bytes = pickle.dumps(obj)
    base64_bytes = base64.b64encode(message_bytes)
    txt = base64_bytes.decode("ascii")
    return txt


def load_or_build_graph(get_data_frame_func):
    """Either load the graph from disk or cretae a new graph and save to disk"""
    print("load_or_build_graph")
    if os.path.exists(GRAPH_PATH):
        # Load the graph from disk
        with gzip.open(GRAPH_PATH, "rb") as f:
            graph = pickle.load(f)
        print("Knowledge graph loaded from disk.")
        is_graph_new = False
    else:
        print("Graph file not found. Building the knowledge graph...")
        # Build the graph using your existing function
        graph = create_graph_from_data_frame(get_data_frame_func)
        # Save the graph to disk
        with gzip.open(GRAPH_PATH, "wb") as f:
            pickle.dump(graph, f)
        print("Knowledge graph saved to disk.")
        is_graph_new = True
    print(f"Number of edged in load_or_build_graph ({graph.number_of_edges()})")
    return graph, is_graph_new


def create_embedding_from_graph(graph, client):
    """Create embedding from knowledge graph"""
    print("Create embedding from knowledge graph")
    # Prepare triples from the graph
    triples = []

    for u, v, data_edge in graph.edges(data=True):
        relation = data_edge["relation"]

        if relation:
            triples.append((str(u), relation, str(v)))
        else:
            print(
                f"""Warning: Edge ({u}, {v}) does not have a 'relation' attribute
                  and will be skipped."""
            )

    if not triples:
        raise ValueError(
            "The 'triples' list is empty. Please check the graph construction."
        )
    else:
        print(len(triples))

    triples_data_frame = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    print(triples_data_frame.head())

    # Extract unique entities from the triples
    entities = set(triples_data_frame["head"]).union(set(triples_data_frame["tail"]))

    entity_embeddings, id_to_entity = get_entity_embeddings(entities, client)

    # Create FAISS index (this part remains similar)
    embedding_dimension = entity_embeddings.shape[1]
    print(f"Creating FAISS index with embedding dimension: {embedding_dimension}")
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(entity_embeddings)

    return index, id_to_entity


def get_entity_embeddings(entities, client, model="text-embedding-ada-002"):
    """Get embeddings for a list of entities using OpenAI API in batches."""

    max_context_length = 8191  # For text-embedding-ada-002
    entity_embeddings = []
    id_to_entity = {}
    batch = []
    batch_token_count = 0
    encoding = tiktoken.encoding_for_model(model)

    entities = list(entities)  # Ensure it's a list for indexing
    print(f'There are a total of ({len(entities)}) entities')

    for idx, entity in enumerate(entities):
        entity_str = str(entity)
        tokens = encoding.encode(entity_str)
        token_count = len(tokens)

        # If adding this input exceeds the max context length or batch size limit, process the
        # current batch
        if (batch_token_count + token_count > max_context_length) or (
            len(batch) >= BATCH_SIZE_LIMIT
        ):
            # Process the current batch
            # print(
            #     f"Sending a batch of ({len(batch)} entities with ({batch_token_count}) tokens.)"
            # )
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [data.embedding for data in response.data]
            for i, embedding in enumerate(batch_embeddings):
                global_idx = idx - len(batch) + i
                entity_embeddings.append(embedding)
                id_to_entity[global_idx] = batch[i]
            # Reset batch
            batch = []
            batch_token_count = 0

        # Add current input to batch
        batch.append(entity_str)
        batch_token_count += token_count
        if (idx % 10_000) == 0:
            print(f'Finished processing entity ({idx})')

    # Process any remaining inputs in the last batch
    if batch:
        print(
            f"Sending a batch of ({len(batch)} entities with ({batch_token_count}) tokens.)"
        )
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [data.embedding for data in response.data]
        for i, embedding in enumerate(batch_embeddings):
            global_idx = len(entity_embeddings)
            entity_embeddings.append(embedding)
            id_to_entity[global_idx] = batch[i]

    # Convert embeddings to NumPy array
    entity_embeddings = np.array(entity_embeddings, dtype="float32")
    return entity_embeddings, id_to_entity


def get_data_frame_from_csv():
    """Load a data from from a CSV file"""
    # Load the CSV data
    print("get_data_frame_from_csv")
    data = pd.read_csv(FILE_PATH, encoding=FILE_ENCODING)

    # Fill missing values
    na_fill_values = {
        "Contact Date": "1/1/2000",
        "Reference Nbr": "9999",
        "Department": "No Department",
        "Origin": "No Origin",
        "Serial Nbr": "No Serial Number",
        "Model Number": "No Model",
        "Horsepower Group": "No Horsepower Group",
        "Product Item Description": "No Item Description",
        "Status": "No Status",
        "Type": "No Type",
        "Reason Code": "No Reason Code",
        "Reason Code Desc": "No Reason Code Desc",
        "Cylinders": "No Cylinders",
        "Ora Warr Family Desc": "No Ora Warr Family Desc",
        "Dealer OEM Number": 0,
        "Dealer OEM Name": "No Name",
        "Part Code Desc": "No Code",
        "Fail Code Desc": "No Fail Code Desc",
        "Text": "No Text",
    }

    data.fillna(value=na_fill_values, inplace=True)

    print(f"DataFrame Shape in get_data_frame_from_csv: ({data.shape})")
    return data


def create_graph_from_data_frame(data_frame_func):
    """Create knowledge graph from passed in data frame"""
    print("create_graph_from_data_frame")
    data = data_frame_func()
    print(f"DataFrame Shape in create_graph_from_data_frame {data.shape}")
    graph = nx.MultiDiGraph()

    for _, row in data.iterrows():
        # Extract entities
        contact_date = row["Contact Date"]
        reference_nbr = row["Reference Nbr"]
        department = row["Department"]
        origin = row["Origin"]
        serial_nbr = row["Serial Nbr"]
        model_number = row["Model Number"]
        horsepower_group = row["Horsepower Group"]
        product_desc = row["Product Item Description"]
        status = row["Status"]
        contact_type = row["Type"]
        reason_code = row["Reason Code"]
        reason_code_desc = row["Reason Code Desc"]
        cylinders = row["Cylinders"]
        ora_warrenty_family_desc = row["Ora Warr Family Desc"]
        dealer_oem_number = row["Dealer OEM Number"]
        dealer_name = row["Dealer OEM Name"]
        part_code_desc = row["Part Code Desc"]
        fail_code_desc = row["Fail Code Desc"]
        text = row["Text"]

        # Add nodes
        graph.add_node(
            product_desc,
            label="Product",
            model_number=model_number,
            horsepower_group=horsepower_group,
            serial_nbr=serial_nbr,
            department=department,
            cylinders=cylinders,
            ora_warrenty_family_desc=ora_warrenty_family_desc,
        )

        graph.add_node(
            reference_nbr,
            label="Interaction",
            contact_date=contact_date,
            origin=origin,
            status=status,
            contact_type=contact_type,
        )

        graph.add_node(reason_code_desc, label="Issue", reason_code=reason_code)
        graph.add_node(dealer_name, label="Dealer", dealer_oem_number=dealer_oem_number)
        graph.add_node(part_code_desc, label="Part", fail_code_desc=fail_code_desc)
        graph.add_node(text, label="Comment")

        # Add edges
        graph.add_edge(reference_nbr, product_desc, relation="interation_product")
        graph.add_edge(product_desc, part_code_desc, relation="product_part")
        graph.add_edge(reference_nbr, reason_code_desc, relation="interation_issue")
        graph.add_edge(product_desc, dealer_name, relation="product_dealter")
        graph.add_edge(reference_nbr, text, relation="interation_text")

    print(
        f"Number of edged in create_graph_from_data_frame ({graph.number_of_edges()})"
    )

    return graph


def get_text_embedding(text, client, model="text-embedding-ada-002"):
    """Function to get the embedding of a query using text embeddings"""
    # Ensure text is a valid string
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid text input: {text}")

    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding


def get_relevant_entities(query, index, id_to_entity, client, k=5):
    """Function to retrieve relevant entities from the knowledge graph"""
    # Get the text embedding of the query
    query_embedding = get_text_embedding(query, client)  # astype("float32")
    # Since the knowledge graph embeddings are in the same space, we can search directly
    _, indices = index.search(np.array([query_embedding]), k)
    # Retrieve entity names
    relevant_entities = []
    for idx in indices[0]:
        if idx in id_to_entity.keys():
            print(f'Index ({idx}) exists')
            relevant_entities.append(id_to_entity[idx])
        else:
            print(f'Index ({idx}) doesn\'t exist')

    #  relevant_entities = [id_to_entity[idx] for idx in indices[0]]
    return relevant_entities


def create_prompt_with_kg(query, graph, index, id_to_entity, client):
    """Function to create a prompt using knowledge graph information"""
    relevant_entities = get_relevant_entities(query, index, id_to_entity, client)
    context = ""
    for entity in relevant_entities:
        # Extract node attributes
        node_data = graph.nodes[entity]
        node_info = f"Entity: {entity}, Attributes: {node_data}"
        # Get connected edges and nodes
        connections = graph.edges(entity, data=True)
        for _, neighbor, edge_data in connections:
            relation = edge_data["relation"]
            context += f"{entity} {relation} {neighbor}\n"
        context += f"{node_info}\n"

    prompt = f"""
    As a knowledgeable customer service assistant of Mercury Marine Boats,
    use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    print(prompt)
    return prompt.strip()


def generate_llm_response(prompt, client):
    """Function to generate a response using OpenAI's GPT-3.5 Turbo model"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    print(obj_to_txt(response))
    return response.choices[0].message.content


def get_final_response(query, graph, index, id_to_entity, client):
    """Function to get the final response"""
    prompt = create_prompt_with_kg(query, graph, index, id_to_entity, client)
    response = generate_llm_response(prompt, client)
    return response


def load_index_and_mappings(graph, client, is_new_graph):
    """Function to load index and mappings"""
    print("load_index_and_mappings")
    if not is_new_graph and (
        os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH)
    ):
        # Load FAISS index
        index = faiss.read_index(INDEX_PATH)
        print("FAISS index loaded from disk.")

        # Load id_to_entity mapping
        with open(MAPPING_PATH, "rb") as f:
            id_to_entity = pickle.load(f)
        print("id_to_entity mapping loaded from disk.")
    else:
        print("Index or mappings not found on disk. Building from scratch...")
        # Code to build the index and mappings
        index, id_to_entity = create_embedding_from_graph(graph, client)
        # Save them for future use
        faiss.write_index(index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(id_to_entity, f)
        print("Index and mappings saved to disk.")
    return index, id_to_entity


open_ai_client = OpenAI(api_key=OPEN_API_KEY)
graph, is_built_graph = load_or_build_graph(get_data_frame_from_csv)
calllog_idex, calllog_id_to_entity = load_index_and_mappings(
    graph, open_ai_client, is_built_graph
)

if __name__ == "__main__":
    while True:
        user_request = input("Enter your question (or 'q' to quit): ")
        if user_request.lower() == "q":
            print("Goodbye!")
            break
        answer = get_final_response(
            user_request, graph, calllog_idex, calllog_id_to_entity, open_ai_client
        )
        print("\nAssistant:\n", answer)
