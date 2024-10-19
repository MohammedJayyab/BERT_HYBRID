import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
from torch.cuda.amp import autocast
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import lil_matrix
import csv
import os
import pickle

from pretty_table import print_formatted_recommendations, export_recommendations_as_json, save_recommendations_to_json, save_recommendations_to_csv

class HybridRecommendationSystem:
    def __init__(self, product_file, transaction_file, interaction_weights=None, blend_weights=None):
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

        # Customizable interaction weights
        self.interaction_weights = interaction_weights if interaction_weights else {'clicked': 0, 'added': 0, 'purchased': 1}

        # Hyperparameters to blend collaborative and content-based features
        self.blend_weights = blend_weights if blend_weights else {'content': 0.5, 'collaborative': 0.5}

        self.w1 = 0.2  # Interaction Weight
        self.w2 = 0.4  # Content Similarity
        self.w3 = 0.4  # Collaborative Similarity

    def prepare_product_documents(self):
        self.products_df['document'] = self.products_df['product_name']
        print("Sample documents for embedding:")
        print(self.products_df['document'].head())

    def generate_bert_embeddings(self, batch_size=32, embeddings_file='model/bert_embeddings.pkl'):
        if os.path.exists(embeddings_file):
            print("Loading BERT embeddings from file...")
            with open(embeddings_file, 'rb') as f:
                self.product_vectors = pickle.load(f)
            print("BERT embeddings loaded successfully.")
            return

        print("Generating BERT embeddings for product documents...")

        documents = self.products_df['document'].tolist()
        embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        inputs = self.tokenizer(documents, return_tensors='pt', truncation=True, padding=True, max_length=512)

        def process_batch(i):
            batch_inputs = {key: value[i:i + batch_size].to(device) for key, value in inputs.items()}
            with torch.no_grad():
                with autocast():
                    outputs = self.model(**batch_inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, i) for i in range(0, len(documents), batch_size)]
            for future in tqdm(futures, desc="Processing batches"):
                embeddings.append(future.result())

        if len(embeddings) > 0:
            self.product_vectors = np.vstack(embeddings)
            print("BERT embeddings generated successfully.")
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.product_vectors, f)
            print("BERT embeddings saved.")
        else:
            print("Error: No embeddings were generated.")

    def compute_similarity_matrix(self):
        try:
            self.product_vectors = pd.DataFrame(self.product_vectors).apply(pd.to_numeric, errors='coerce').values
            print(f"Product vectors converted to numeric format with shape: {self.product_vectors.shape}")
        except Exception as e:
            print(f"Error during conversion of product vectors to numeric: {e}")
            return

        if self.product_vectors.size == 0:
            print("Error: product_vectors is empty.")
            return

        if np.isnan(self.product_vectors).any():
            nan_indices = np.argwhere(np.isnan(self.product_vectors))
            print("NaN values found in product vectors at the following locations:")
            for row, col in nan_indices:
                print(f"NaN at row {row}, column {col} (Product ID: {self.products_df.iloc[row]['product_id']})")
            self.product_vectors = np.nan_to_num(self.product_vectors)

        if self.product_vectors.size == 0 or self.product_vectors.shape[1] == 0:
            print("Error: product_vectors is invalid.")
            return

        # Using cosine_similarity from scikit-learn
        print("Computing product-to-product cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.product_vectors)
        
        # Normalize the similarity matrix to be between 0 and 1 using Min-Max normalization
        self.similarity_matrix = self.min_max_scaler.fit_transform(self.similarity_matrix)
        print("Similarity matrix computed and normalized successfully.")

    def construct_user_product_graph(self):
        user_nodes = set()
        product_nodes = set()

        print("Processing product nodes...")
        for _, row in tqdm(self.products_df.iterrows(), total=len(self.products_df), desc="Products"):
            product_node = f"product_{row['product_id']}"
            if product_node not in product_nodes:
                self.graph.add_node(product_node, type='product')
                product_nodes.add(product_node)

        print("Processing user-product interactions...")
        for _, row in tqdm(self.transactions_df.iterrows(), total=len(self.transactions_df), desc="Transactions"):
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type']
            quantity = row.get('quantity', 1)
            base_weight = self.interaction_weights.get(interaction_type, 1)
            weight = base_weight * np.log(quantity + 1)

            if user_node not in user_nodes:
                self.graph.add_node(user_node, type='user')
                user_nodes.add(user_node)

            self.graph.add_edge(user_node, product_node, weight=weight, interaction=interaction_type, quantity=quantity)

        print("User-product graph construction completed.")

    def normalize_edge_weights(self):
        all_weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
        sigmoid_weights = 1 / (1 + np.exp(-np.array(all_weights)))

        for idx, (u, v, d) in enumerate(self.graph.edges(data=True)):
            self.graph[u][v]['weight'] = sigmoid_weights[idx]

    def compute_combined_similarity(self):
        print("Computing combined similarity matrix...")
        num_products = len(self.products_df)
        num_users = len(self.transactions_df['customer_id'].unique())
        self.combined_similarity_matrix = np.zeros((num_products, num_products))

        product_to_index = {f"product_{row['product_id']}": idx for idx, row in self.products_df.iterrows()}
        user_to_index = {f"user_{customer_id}": idx for idx, customer_id in enumerate(self.transactions_df['customer_id'].unique())}

        adjacency_matrix = lil_matrix((num_users, num_products))
        for _, row in tqdm(self.transactions_df.iterrows(), desc="Building Adjacency Matrix", total=len(self.transactions_df)):
            user_idx = user_to_index[f"user_{row['customer_id']}"]
            product_idx = product_to_index[f"product_{row['product_id']}"]
            adjacency_matrix[user_idx, product_idx] = 1

        adjacency_matrix = adjacency_matrix.tocsr()
        product_interaction_matrix = adjacency_matrix.T @ adjacency_matrix
        product_interaction_matrix.setdiag(0)
        print("Collaborative similarities computed.")

    def multi_hop_recommendation(self, user_id, hop=2, top_n=5, exclude_purchased=False):
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        recommendations = {}
        purchased_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph[user_node][neighbor]['interaction'] == 'purchased'
        }

        neighbors = set(self.graph.neighbors(user_node))
        for neighbor in neighbors:
            if hop == 1:
                if self.graph.nodes[neighbor]['type'] == 'product':
                    interaction_weight = self.graph[user_node][neighbor]['weight']
                    product_id = neighbor.split('_')[1]

                    product_row = self.products_df[self.products_df['product_id'] == product_id]
                    if product_row.empty:
                        continue

                    product_index = product_row.index[0]
                    content_similarity = self.similarity_matrix[product_index, product_index]
                    score = (self.w1 * interaction_weight + self.w2 * content_similarity)
                    recommendations[neighbor] = score
            else:
                second_hop_neighbors = set(self.graph.neighbors(neighbor))
                for second_neighbor in second_hop_neighbors:
                    if self.graph.nodes[second_neighbor]['type'] == 'user' and second_neighbor != user_node:
                        third_hop_products = set(self.graph.neighbors(second_neighbor))
                        for product in third_hop_products:
                            if self.graph.nodes[product]['type'] == 'product' and product not in neighbors:
                                interaction_weight = self.graph[second_neighbor][product]['weight']
                                product_id = product.split('_')[1]

                                product_row = self.products_df[self.products_df['product_id'] == product_id]
                                if product_row.empty:
                                    continue

                                product_index = product_row.index[0]
                                content_similarity = self.similarity_matrix[product_index, product_index]
                                score = (self.w1 * interaction_weight + self.w2 * content_similarity)
                                recommendations[product] = recommendations.get(product, 0) + score

        if exclude_purchased:
            recommendations = {prod: score for prod, score in recommendations.items() if prod not in purchased_products}

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [(self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['category_name'].values[0],
                self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_id'].values[0],
                self.products_df[self.products_df['product_id'] == item[0].split('_')[1]]['product_name'].values[0],
                item[1],
                'collaborative') for item in sorted_recommendations]
    
    def recommend_content_based_products(self, user_id, n=5, similarity_penalty_threshold=0.8):
        # Recommend new products that the user has not interacted with
        user_node = f"user_{user_id}"
        
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        # Step 1: Precompute product lookup and product indices for faster access
        product_lookup = self.products_df.set_index('product_id').to_dict(orient='index')
        
        product_index_map = {p_id: idx for idx, p_id in enumerate(self.products_df['product_id'])}

        # Get products the user has interacted with
        interacted_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph.nodes[neighbor]['type'] == 'product'
        }

        interacted_product_ids = {p.split('_')[1] for p in interacted_products}
        interacted_product_names = {product_lookup[p_id]['product_name'] for p_id in interacted_product_ids if p_id in product_lookup}

        all_product_ids = set(self.products_df['product_id'].tolist())
        non_interacted_product_ids = all_product_ids - interacted_product_ids

        # Filter out products already interacted with
        filtered_non_interacted_products = [
            p_id for p_id in non_interacted_product_ids if p_id in product_lookup
        ]

        # If no interacted products found, return empty list
        if not interacted_product_ids:
            print("No interacted products found for this user.")
            return []

        # Get indices of interacted products for similarity comparison
        interacted_indices = [
            product_index_map[p_id] for p_id in interacted_product_ids if p_id in product_index_map
        ]

        recommendations = []

        # Step 2: Calculate similarity and gather recommendations
        for product_id in filtered_non_interacted_products:
            product_info = product_lookup[product_id]
            product_name = product_info['product_name']
            product_index = product_index_map.get(product_id)

            if product_index is None:
                continue

            # Calculate similarity with interacted products
            content_similarity = np.max(self.similarity_matrix[product_index, interacted_indices])

            # Add the filter type back into the recommendation tuple
            recommendations.append((product_info['category_name'], product_id, product_name, content_similarity, 'content-based'))

        # Step 3: Filter out products that are too similar to each other
        filtered_recommendations = []
        product_indices_in_recs = []

        for category_name, product_id, product_name, score, filter_type in recommendations:
            product_index = product_index_map[product_id]

            # Compare this product to already selected products
            too_similar = False
            for selected_index in product_indices_in_recs:
                similarity = self.similarity_matrix[product_index, selected_index]
                if similarity > similarity_penalty_threshold:
                    too_similar = True
                    break

            if not too_similar:
                filtered_recommendations.append((category_name, product_id, product_name, score, filter_type))
                product_indices_in_recs.append(product_index)

            # Stop when we have enough recommendations
            if len(filtered_recommendations) >= n:
                break

        # Step 4: Sort the recommendations by score (if necessary)
        filtered_recommendations.sort(key=lambda x: x[3], reverse=True)

        # Print content-based recommendations
        print(f"Content-based Recommendations for user '{user_id}':")
        print("Category Name | Product ID | Product Name | Score | Filter Type")
        for rec in filtered_recommendations:
            print(rec)

        return filtered_recommendations



    def recommend_category_based_products(self, user_id, recommendations, n=3):
        user_node = f"user_{user_id}"

        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        product_lookup = self.products_df.set_index('product_id').to_dict(orient='index')
        user_transactions = self.transactions_df[self.transactions_df['customer_id'] == user_id]
        interacted_product_ids = set(user_transactions['product_id'])
        interacted_categories = set(self.products_df[self.products_df['product_id'].isin(interacted_product_ids)]['category_name'])
        recommended_categories = {rec[0] for rec in recommendations}

        non_used_categories = interacted_categories - recommended_categories

        if not non_used_categories:
            return []

        category_recommendations = []

        for category in non_used_categories:
            products_in_category = self.products_df[self.products_df['category_name'] == category]
            if not products_in_category.empty:
                random_product = products_in_category.sample(n=1, random_state=42).iloc[0]
                category_recommendations.append((
                    random_product['category_name'],
                    random_product['product_id'],
                    random_product['product_name'],
                    1.0,  # Fixed score for category-based recommendations
                    'category-based'
                ))

        category_recommendations.sort(key=lambda x: (-x[3], x[0])[:n])
        return category_recommendations

    def blend_user_and_content_and_category_recommendations(self, user_recommendations, content_recommendations, category_recommendation, total_n=10, user_ratio=0.3, content_ratio=0.3, category_ratio=0.4):
        num_user_recommendations = int(round(total_n * user_ratio))
        num_content_recommendations = int(round(total_n * content_ratio))
        num_category_recommendations = int(round(total_n * category_ratio))

        total_recommendations = num_user_recommendations + num_content_recommendations + num_category_recommendations

        if total_recommendations > total_n:
            excess = total_recommendations - total_n
            num_category_recommendations -= excess
        elif total_recommendations < total_n:
            deficit = total_n - total_recommendations
            num_category_recommendations += deficit

        user_recommendations_trimmed = user_recommendations[:num_user_recommendations]

        user_recommendation_ids = {recommendation[1] for recommendation in user_recommendations_trimmed}
        content_recommendations_filtered = [
            recommendation for recommendation in content_recommendations
            if recommendation[1] not in user_recommendation_ids
        ]

        content_recommendations_trimmed = content_recommendations_filtered[:num_content_recommendations]
        category_recommendation_trimmed = category_recommendation[:num_category_recommendations]

        blended_recommendations = user_recommendations_trimmed + content_recommendations_trimmed + category_recommendation_trimmed
        blended_recommendations = sorted(blended_recommendations, key=lambda x: x[2], reverse=True)

        return blended_recommendations

    def run_recommendation_pipeline(self):
        self.prepare_product_documents()
        self.generate_bert_embeddings()
        self.compute_similarity_matrix()
        self.construct_user_product_graph()
        self.normalize_edge_weights()
        self.compute_combined_similarity()


if __name__ == "__main__":
    def print_time(prefix):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{prefix} {formatted_time}")

    def read_customer_ids(file_path):
        customer_ids = []
        try:
            with open(file_path, mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for row in csv_reader:
                    if row:
                        customer_ids.append(row[0].strip('"'))
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
        except IOError:
            print(f"Error: An error occurred while reading the file at {file_path}.")
        return customer_ids

    print_time("Starting at")
    start_time = time.time()

    recommender = HybridRecommendationSystem('data/ml_products.csv', 'data/ml_transactions.csv')
    recommender.run_recommendation_pipeline()

    customer_ids = read_customer_ids('data/ml_customers.csv')

    all_customer_recommendations = {}

    for customer_id in tqdm(customer_ids, desc="Processing customers"):
        user_recommendations = recommender.multi_hop_recommendation(customer_id, hop=2, top_n=5, exclude_purchased=True)
        user_recommendations_trimmed = [(category_name, product_id, score, filter_type) for category_name, product_id, _, score, filter_type in user_recommendations]

        content_recommendations = recommender.recommend_content_based_products(customer_id, n=15)
        content_recommendations_trimmed = [(category_name, product_id, score, filter_type) for category_name, product_id, _, score, filter_type in content_recommendations]

        category_recommendation = recommender.recommend_category_based_products(customer_id, user_recommendations_trimmed + content_recommendations, n=5)
        category_recommendations_trimmed = [(category_name, product_id, score, filter_type) for category_name, product_id, _, score, filter_type in category_recommendation]

        blended_recommendations = recommender.blend_user_and_content_and_category_recommendations(
            user_recommendations_trimmed,
            content_recommendations_trimmed,
            category_recommendations_trimmed,
            total_n=10,
            user_ratio=0.3,
            content_ratio=0.3,
            category_ratio=0.4
        )

        customer_recommendations_json = export_recommendations_as_json(blended_recommendations, recommender.products_df, customer_id)
        all_customer_recommendations.update(customer_recommendations_json)

    save_recommendations_to_json('output/customer_recommendations.json', all_customer_recommendations)

    end_time = time.time()
    print_time("Ending at")
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds.")
