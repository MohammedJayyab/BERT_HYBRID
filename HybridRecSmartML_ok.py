import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
import torch
from torch.cuda.amp import autocast  # Mixed precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix  # For sparse matrix representation



# Import the print_formatted_recommendations function from pretty_table.py
from pretty_table import print_formatted_recommendations
from pretty_table import export_recommendations_as_json
from pretty_table import save_recommendations_to_json


class HybridRecommendationSystem:
    def __init__(self, product_file, transaction_file, interaction_weights=None, blend_weights=None):
        # Load data and ensure product_id and customer_id are strings
        self.products_df = pd.read_csv(product_file, dtype={'product_id': str})
        self.transactions_df = pd.read_csv(transaction_file, dtype={'product_id': str, 'customer_id': str})
        self.graph = nx.Graph()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.product_vectors = None
        self.similarity_matrix = None
        self.combined_similarity_matrix = None
        self.min_max_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()

        # Customizable interaction weights for user behavior (clicked, added, purchased)
        self.interaction_weights = interaction_weights if interaction_weights else {'clicked': 0, 'added': 0, 'purchased': 1}
        
        # Hyperparameters to blend collaborative and content-based features
        self.blend_weights = blend_weights if blend_weights else {'content': 0.5, 'collaborative': 0.5}

        # Weight parameters for interaction weight, content similarity, and collaborative similarity
        self.w1 = 0.2  # Interaction Weight
        self.w2 = 0.4  # Content Similarity
        self.w3 = 0.4  # Collaborative Similarity

    def prepare_product_documents(self):
        # Concatenate product name, category, and description
        self.products_df['document'] = self.products_df['product_name']  
        # Print the first few documents to check if they are correctly generated
        print("Sample documents for embedding:")
        print(self.products_df['document'].head())



    def generate_bert_embeddings(self, batch_size=32):
        # Generate BERT embeddings for product documents in batches
        print("Generating BERT embeddings for product documents...")

        documents = self.products_df['document'].tolist()  # Get all documents as a list

        # If documents are empty, raise an error early
        if len(documents) == 0:
            print("Error: No documents found to generate embeddings.")
            return

        embeddings = []

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        if torch.cuda.is_available():
            print("Using GPU for BERT model.")
        else:
            print("Using CPU for BERT model.")

        # Pre-tokenize all documents to avoid tokenization overhead in each batch
        print("Pre-tokenizing all documents...")
        inputs = self.tokenizer(documents, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # Initialize tqdm progress bar for batch processing
        progress_bar = tqdm(range(0, len(documents), batch_size), desc="Processing batches")

        # Process documents in batches to improve performance
        for i in progress_bar:
            # Create a batch of inputs by slicing each tensor (e.g., input_ids, attention_mask)
            batch_inputs = {key: value[i:i + batch_size].to(device) for key, value in inputs.items()}

            # Get embeddings without mixed precision as a test
            with torch.no_grad():
                outputs = self.model(**batch_inputs)

            # Get the mean of the last hidden state and keep on GPU for faster processing
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Check if any embeddings are generated
            if batch_embeddings.size == 0:
                print(f"Error: No embeddings generated for batch {i // batch_size + 1}.")
                continue

            embeddings.append(batch_embeddings)

        # Combine all embeddings into a single array
        if len(embeddings) > 0:
            self.product_vectors = np.vstack(embeddings)
            print("BERT embeddings generated successfully.")
        else:
            print("Error: No embeddings were generated for any document.")

    # def compute_similarity_matrix(self):
    #     # Compute product-to-product cosine similarity matrix
    #     print("Computing product-to-product cosine similarity matrix...")
    #     self.similarity_matrix = cosine_similarity(self.product_vectors)
    #     # Normalize the similarity matrix to be between 0 and 1 using Min-Max normalization
    #     self.similarity_matrix = self.min_max_scaler.fit_transform(self.similarity_matrix)
    def compute_similarity_matrix(self):
        # Ensure embeddings are numeric
        try:
            # First convert product_vectors to a numeric type, coerce invalid values to NaN
            self.product_vectors = pd.DataFrame(self.product_vectors).apply(pd.to_numeric, errors='coerce').values
            print(f"Product vectors converted to numeric format with shape: {self.product_vectors.shape}")
        except Exception as e:
            print(f"Error during conversion of product vectors to numeric: {e}")
            return

        # Check if product_vectors is empty
        if self.product_vectors.size == 0:
            print("Error: product_vectors is empty. No valid embeddings were generated.")
            return

        # Check for NaN values
        if np.isnan(self.product_vectors).any():
            # Find and log the exact locations of NaN values
            nan_indices = np.argwhere(np.isnan(self.product_vectors))  # Get indices of NaNs

            print("NaN values found in product vectors at the following locations:")
            for row, col in nan_indices:
                print(f"NaN found at row {row}, column {col} (Product ID: {self.products_df.iloc[row]['product_id']})")

            print("Replacing NaN values with 0 in the embeddings.")
            self.product_vectors = np.nan_to_num(self.product_vectors)  # Replace NaN with 0
        
        # Double-check if product_vectors is still valid after NaN replacement
        if self.product_vectors.size == 0 or self.product_vectors.shape[1] == 0:
            print("Error: product_vectors is invalid or empty after NaN handling.")
            return

        # Compute product-to-product cosine similarity matrix
        print("Computing product-to-product cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.product_vectors)
        
        # Normalize the similarity matrix to be between 0 and 1 using Min-Max normalization
        self.similarity_matrix = self.min_max_scaler.fit_transform(self.similarity_matrix)
        print("Similarity matrix computed and normalized successfully.")

    def construct_user_product_graph(self):
        # Initialize sets to track added user and product nodes
        user_nodes = set()
        product_nodes = set()

        # Add product nodes to the graph in batch
        print("Processing product nodes...")
        for _, row in tqdm(self.products_df.iterrows(), total=len(self.products_df), desc="Products"):
            product_node = f"product_{row['product_id']}"
            if product_node not in product_nodes:
                self.graph.add_node(product_node, type='product')
                product_nodes.add(product_node)

        # Add user-product interactions to the graph
        print("Processing user-product interactions...")
        for _, row in tqdm(self.transactions_df.iterrows(), total=len(self.transactions_df), desc="Transactions"):
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type']
            quantity = row.get('quantity', 1)  # Default to 1 if quantity is not present
            base_weight = self.interaction_weights.get(interaction_type, 1)
            weight = base_weight * np.log(quantity + 1)  # Final weight is based on interaction type and quantity with logarithmic scaling

            # Only add user node if it hasn't been added yet
            if user_node not in user_nodes:
                self.graph.add_node(user_node, type='user')
                user_nodes.add(user_node)

            # Add an edge with interaction type, weight, and quantity
            self.graph.add_edge(user_node, product_node, weight=weight, interaction=interaction_type, quantity=quantity)

        print("User-product graph construction completed.")

    def normalize_edge_weights(self):
        # Normalize edge weights in the user-product graph to be between 0 and 1
        all_weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
        weights_array = np.array(all_weights).reshape(-1, 1)
        normalized_weights = self.min_max_scaler.fit_transform(weights_array).flatten()

        # Update graph with normalized weights
        for idx, (u, v, d) in enumerate(self.graph.edges(data=True)):
            self.graph[u][v]['weight'] = normalized_weights[idx]

    def compute_combined_similarity(self):
        print("Computing combined similarity matrix...")

        num_products = len(self.products_df)
        num_users = len(self.transactions_df['customer_id'].unique())

        # Initialize the combined similarity matrix
        self.combined_similarity_matrix = np.zeros((num_products, num_products))

        # Step 1: Create the product-user adjacency matrix
        product_to_index = {f"product_{row['product_id']}": idx for idx, row in self.products_df.iterrows()}
        user_to_index = {f"user_{customer_id}": idx for idx, customer_id in enumerate(self.transactions_df['customer_id'].unique())}

        # Create a sparse matrix for product-user interactions (users as rows, products as columns)
        adjacency_matrix = lil_matrix((num_users, num_products))

        # Fill the adjacency matrix with interactions
        for _, row in tqdm(self.transactions_df.iterrows(), desc="Building Adjacency Matrix", total=len(self.transactions_df)):
            user_idx = user_to_index[f"user_{row['customer_id']}"]
            product_idx = product_to_index[f"product_{row['product_id']}"]
            adjacency_matrix[user_idx, product_idx] = 1

        # Convert to CSR format for efficient matrix multiplication
        adjacency_matrix = adjacency_matrix.tocsr()

        # Step 2: Use matrix multiplication to compute the common neighbors (collaborative similarity)
        print("Computing collaborative similarities...")
        product_interaction_matrix = adjacency_matrix.T @ adjacency_matrix  # Matrix multiplication

        # The diagonal contains self-product interactions, set them to 0
        product_interaction_matrix.setdiag(0)
        print("Collaborative similarities computed successfully.")
        # Step 3: Normalize collaborative similarities using Z-score normalization
        #collaborative_similarities = product_interaction_matrix.toarray().reshape(-1, 1)
        #normalized_collaborative_similarities = self.standard_scaler.fit_transform(collaborative_similarities).flatten()
 

    def get_top_n_similar_products(self, product_id, n=5):
        # Retrieve top-N similar products based on the combined similarity matrix
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        product_similarities = self.combined_similarity_matrix[product_idx]
        top_n_indices = np.argsort(product_similarities)[::-1][1:n+1]

        # Create a DataFrame for the top N products with similarity scores
        similar_products = self.products_df.iloc[top_n_indices][['category_name', 'product_id', 'product_name']].copy()
        similar_products['score'] = product_similarities[top_n_indices]
        similar_products['filter_type'] = 'collaborative' 

        return similar_products
    def recommend_new_or_non_interacted_products(self, user_id, n=5, verbose=False):
        # Recommend new products that the user has not interacted with
        user_node = f"user_{user_id}"
        
        # Check if the user node exists in the graph
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        # Step 1: Precompute product lookup and product indices for faster access
        product_lookup = self.products_df.set_index('product_id').to_dict(orient='index')
        
        # Precompute indices of all products
        product_index_map = {p_id: idx for idx, p_id in enumerate(self.products_df['product_id'])}

        # Get the products the user has interacted with
        interacted_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph.nodes[neighbor]['type'] == 'product'
        }

        # Create sets of product IDs and names that the user has interacted with
        interacted_product_ids = {p.split('_')[1] for p in interacted_products}
        interacted_product_names = {product_lookup[p_id]['product_name'] for p_id in interacted_product_ids if p_id in product_lookup}

        # Step 2: Filter out non-interacted products early to reduce later computations
        all_product_ids = set(self.products_df['product_id'].tolist())
        non_interacted_product_ids = all_product_ids - interacted_product_ids

        # Step 3: Use set operations to filter out duplicates (case-insensitive)
        seen_product_names = set()
        filtered_non_interacted_products = []

        for product_id in non_interacted_product_ids:
            if product_id in product_lookup:
                product_name = product_lookup[product_id]['product_name']
                if product_name not in seen_product_names:
                    filtered_non_interacted_products.append(product_id)
                    seen_product_names.add(product_name)

        recommendations = []
        skipped_products_count = 0  # Track how many products were skipped

        # Step 4: Precompute interacted product indices to avoid repeated lookups
        interacted_indices = [
            product_index_map[p_id]
            for p_id in interacted_product_ids if p_id in product_index_map
        ]

        if not interacted_indices:
            print("No interacted products found for this user.")
            return []

        # Step 5: Vectorize similarity calculation using numpy for faster computation
        for product_id in filtered_non_interacted_products:
            if product_id in product_lookup:
                product_info = product_lookup[product_id]
                product_name = product_info['product_name']

                # Skip if the product_id or product_name is in the interacted products set
                if product_id in interacted_product_ids or product_name in interacted_product_names:
                    if verbose:
                        print(f"Skipping already interacted product '{product_name}' with ID '{product_id}'.")
                    continue  # Skip this product if it was already interacted with

                product_index = product_index_map.get(product_id)
                if product_index is None:
                    skipped_products_count += 1
                    if verbose:
                        print(f"Product with ID '{product_id}' not found in product index map.")
                    continue

                # Calculate similarity in bulk with vectorized operation
                content_similarity = np.max(self.similarity_matrix[product_index, interacted_indices])

                # Only content similarity is used as there's no interaction or collaborative data for new products
                score = content_similarity
                
                recommendations.append((product_info['category_name'],
                                        product_id,
                                        product_name,
                                        score,
                                        'content-based'))

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations, key=lambda x: x[3], reverse=True)[:n]
        
        # Provide summary if products were skipped
        if skipped_products_count > 0:
            print(f"Skipped {skipped_products_count} products that were not found in products_df.")

        return sorted_recommendations


    def multi_hop_recommendation___old(self, user_id, hop=2, top_n=5, exclude_purchased=False):
        # Generate multi-hop recommendations
        user_node = f"user_{user_id}"
        
        # Check if the user node exists in the graph
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        recommendations = {}

        # Get products the user has "purchased"
        purchased_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph[user_node][neighbor]['interaction'] == 'purchased'
        }

        # Get the direct neighbors of the user (first-hop)
        neighbors = set(self.graph.neighbors(user_node))

        for neighbor in neighbors:
            if hop == 1:
                if self.graph.nodes[neighbor]['type'] == 'product':
                    interaction_weight = self.graph[user_node][neighbor]['weight']
                    product_index = self.products_df[self.products_df['product_id'] == neighbor.split('_')[1]].index[0]
                    content_similarity = self.similarity_matrix[product_index, product_index]
                    collaborative_similarity = 0  # Since hop is 1, there's no collaborative similarity involved
                    
                    # Final scoring function incorporating interaction weight, content similarity, and collaborative similarity
                    score = (self.w1 * interaction_weight +
                             self.w2 * content_similarity +
                             self.w3 * collaborative_similarity)
                    recommendations[neighbor] = score
            else:
                # Get second-hop neighbors (products connected to the user's first-hop products)
                second_hop_neighbors = set(self.graph.neighbors(neighbor))
                
                for second_neighbor in second_hop_neighbors:
                    if self.graph.nodes[second_neighbor]['type'] == 'user' and second_neighbor != user_node:
                        # Get products connected to this second-hop user
                        third_hop_products = set(self.graph.neighbors(second_neighbor))
                        for product in third_hop_products:
                            if self.graph.nodes[product]['type'] == 'product' and product not in neighbors:
                                interaction_weight = self.graph[second_neighbor][product]['weight']
                                product_index = self.products_df[self.products_df['product_id'] == product.split('_')[1]].index[0]
                                content_similarity = self.similarity_matrix[product_index, product_index]
                                collaborative_similarity = self.combined_similarity_matrix[product_index, product_index]
                                
                                # Final scoring function incorporating interaction weight, content similarity, and collaborative similarity
                                score = (self.w1 * interaction_weight +
                                         self.w2 * content_similarity +
                                         self.w3 * collaborative_similarity)
                                recommendations[product] = recommendations.get(product, 0) + score

        # If exclude_purchased is True, filter out products the user has already purchased
        if exclude_purchased:
            recommendations = {prod: score for prod, score in recommendations.items() if prod not in purchased_products}

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Extract category_name, product_id, product_name, and score for each recommended product
        return [(self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['category_name'].values[0],
                 self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_id'].values[0],
                 self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_name'].values[0],
                 item[1]) for item in sorted_recommendations]

    def multi_hop_recommendation(self, user_id, hop=2, top_n=5, exclude_purchased=False):
        # Generate multi-hop recommendations
        user_node = f"user_{user_id}"
        
        # Check if the user node exists in the graph
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        recommendations = {}

        # Get products the user has "purchased"
        purchased_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph[user_node][neighbor]['interaction'] == 'purchased'
        }

        # Get the direct neighbors of the user (first-hop)
        neighbors = set(self.graph.neighbors(user_node))

        for neighbor in neighbors:
            if hop == 1:
                if self.graph.nodes[neighbor]['type'] == 'product':
                    interaction_weight = self.graph[user_node][neighbor]['weight']
                    product_id = neighbor.split('_')[1]
                    
                    # Check if the product_id exists in self.products_df
                    product_row = self.products_df[self.products_df['product_id'] == product_id]
                    
                    if product_row.empty:
                        print(f"Product with ID '{product_id}' not found in products_df.")
                        continue
                    
                    product_index = product_row.index[0]
                    content_similarity = self.similarity_matrix[product_index, product_index]
                    collaborative_similarity = 0  # Since hop is 1, there's no collaborative similarity involved
                    
                    # Final scoring function incorporating interaction weight, content similarity, and collaborative similarity
                    score = (self.w1 * interaction_weight +
                            self.w2 * content_similarity +
                            self.w3 * collaborative_similarity)
                    recommendations[neighbor] = score
            else:
                # Get second-hop neighbors (products connected to the user's first-hop products)
                second_hop_neighbors = set(self.graph.neighbors(neighbor))
                
                for second_neighbor in second_hop_neighbors:
                    if self.graph.nodes[second_neighbor]['type'] == 'user' and second_neighbor != user_node:
                        # Get products connected to this second-hop user
                        third_hop_products = set(self.graph.neighbors(second_neighbor))
                        for product in third_hop_products:
                            if self.graph.nodes[product]['type'] == 'product' and product not in neighbors:
                                interaction_weight = self.graph[second_neighbor][product]['weight']
                                product_id = product.split('_')[1]

                                # Check if the product_id exists in self.products_df
                                product_row = self.products_df[self.products_df['product_id'] == product_id]
                                
                                if product_row.empty:
                                    print(f"Product with ID '{product_id}' not found in products_df.")
                                    continue

                                product_index = product_row.index[0]
                                content_similarity = self.similarity_matrix[product_index, product_index]
                                collaborative_similarity = self.combined_similarity_matrix[product_index, product_index]
                                
                                # Final scoring function incorporating interaction weight, content similarity, and collaborative similarity
                                score = (self.w1 * interaction_weight +
                                        self.w2 * content_similarity +
                                        self.w3 * collaborative_similarity)
                                recommendations[product] = recommendations.get(product, 0) + score

        # If exclude_purchased is True, filter out products the user has already purchased
        if exclude_purchased:
            recommendations = {prod: score for prod, score in recommendations.items() if prod not in purchased_products}

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Extract category_name, product_id, product_name, and score for each recommended product
        return [(self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['category_name'].values[0],
                self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_id'].values[0],
                self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_name'].values[0],
                item[1],
                'collaborative') for item in sorted_recommendations]

    def blend_user_and_new_recommendations(self, user_recommendations, new_recommendations, user_ratio=0.6, new_ratio=0.4):
        # Determine the number of items to take from each list
        num_user_recommendations = int(len(user_recommendations) * user_ratio)
        num_new_recommendations = int(len(new_recommendations) * new_ratio)

        # Limit the user and new recommendations to the calculated sizes
        user_recommendations_trimmed = user_recommendations[:num_user_recommendations]

        # Get product IDs that are already in the user recommendations to prevent duplication
        user_recommendation_ids = {recommendation[1] for recommendation in user_recommendations_trimmed}

        # Filter new recommendations to exclude products that are already in user recommendations
        new_recommendations_filtered = [
            recommendation for recommendation in new_recommendations
            if recommendation[1] not in user_recommendation_ids
        ]

        # Limit to the required number of new recommendations
        new_recommendations_trimmed = new_recommendations_filtered[:num_new_recommendations]

        # Concatenate and sort by score
        blended_recommendations = user_recommendations_trimmed + new_recommendations_trimmed
        blended_recommendations = sorted(blended_recommendations, key=lambda x: x[2], reverse=True)

        return blended_recommendations

    def run_recommendation_pipeline(self):
        # Complete pipeline to generate recommendations
        self.prepare_product_documents()
        self.generate_bert_embeddings()
        self.compute_similarity_matrix()
        self.construct_user_product_graph()
        self.normalize_edge_weights()
        self.compute_combined_similarity()

# Example Usage
def read_customer_ids(file_path):
    with open(file_path, 'r') as file:
        customer_ids = [line.strip().strip('"') for line in file.readlines()]  # Strip quotes and whitespace
    return customer_ids

if __name__ == "__main__":
    recommender = HybridRecommendationSystem('data/ml_products.csv', 'data/ml_transactions.csv')
    recommender.run_recommendation_pipeline()

    # Read customer IDs from the input file
    customer_ids = read_customer_ids('input/customer_ids.txt')

    all_customer_recommendations = {}

    for customer_id in customer_ids:
        #print(f"Top 5 recommendations for user '{customer_id}' with two-hop traversal (excluding purchased):")
        user_recommendations = recommender.multi_hop_recommendation(customer_id, hop=2, top_n=10, exclude_purchased=True)

        user_recommendations_trimmed = [(category_name, product_id, score, filter_type) for category_name, product_id, _, score, filter_type in user_recommendations]

        #print_formatted_recommendations(user_recommendations_trimmed, recommender.products_df)

        #print(f"Top 5 new or non-interacted product recommendations for user '{customer_id}':")
        new_recommendations = recommender.recommend_new_or_non_interacted_products(customer_id, n=10)

        new_recommendations_trimmed = [(category_name, product_id, score, filter_type) for category_name, product_id, _, score, filter_type in new_recommendations]

        #print_formatted_recommendations(new_recommendations_trimmed, recommender.products_df)

        # Blend user and new recommendations
        blended_recommendations = recommender.blend_user_and_new_recommendations(
            user_recommendations_trimmed,
            new_recommendations_trimmed,
            user_ratio=0.5,   # 50% from user recommendations
            new_ratio=0.5     # 50% from new recommendations        
        )
        print(f"=> Blended recommendations for user '{customer_id}':\r\n")

        print_formatted_recommendations(blended_recommendations, recommender.products_df)

        # Export recommendations as JSON for this customer
        customer_recommendations_json = export_recommendations_as_json(blended_recommendations, recommender.products_df, customer_id)
        all_customer_recommendations.update(customer_recommendations_json)

    # Save all customer recommendations to a JSON file
    save_recommendations_to_json('output/customer_recommendations.json', all_customer_recommendations)

