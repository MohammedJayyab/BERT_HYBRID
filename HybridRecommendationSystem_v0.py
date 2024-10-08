import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch

# Import the print_formatted_recommendations function from pretty_table.py
from pretty_table import print_formatted_recommendations

class HybridRecommendationSystem:
    def __init__(self, product_file, transaction_file):
        # Load data
        self.products_df = pd.read_csv(product_file)
        self.transactions_df = pd.read_csv(transaction_file)
        self.graph = nx.Graph()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.product_vectors = None
        self.similarity_matrix = None
        self.combined_similarity_matrix = None

    def prepare_product_documents(self):
        # Concatenate product name, category, and description
        self.products_df['document'] = self.products_df['product_name'] + " " + \
                                       self.products_df['category_name'] + " " + \
                                       self.products_df['description']

    def generate_bert_embeddings(self):
        # Generate BERT embeddings for each product document
        def get_bert_embedding(text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()
        
        self.products_df['embedding'] = self.products_df['document'].apply(get_bert_embedding)
        self.product_vectors = np.vstack(self.products_df['embedding'].values)

    def compute_similarity_matrix(self):
        # Compute product-to-product cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.product_vectors)

    def construct_user_product_graph(self):
        # Add product nodes to the graph
        for _, row in self.products_df.iterrows():
            self.graph.add_node(f"product_{row['product_id']}", type='product')

        # Add user-product interactions to the graph
        for _, row in self.transactions_df.iterrows():
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type']
            weight = {'clicked': 1, 'added': 2, 'purchased': 3}[interaction_type]

            if user_node not in self.graph:
                self.graph.add_node(user_node, type='user')
            
            self.graph.add_edge(user_node, product_node, weight=weight, interaction=interaction_type)

    def compute_combined_similarity(self):
        # Combine collaborative similarity with content-based similarity
        num_products = len(self.products_df)
        self.combined_similarity_matrix = np.zeros((num_products, num_products))
        
        for i in range(num_products):
            for j in range(num_products):
                if i != j:
                    product_i = f"product_{self.products_df.iloc[i]['product_id']}"
                    product_j = f"product_{self.products_df.iloc[j]['product_id']}"

                    common_users = len(set(self.graph.neighbors(product_i)) & set(self.graph.neighbors(product_j)))
                    collaborative_similarity = common_users / (len(set(self.graph.neighbors(product_i))) * len(set(self.graph.neighbors(product_j))) + 1e-9)
                    content_similarity = self.similarity_matrix[i, j]

                    self.combined_similarity_matrix[i, j] = collaborative_similarity + content_similarity

    def get_top_n_similar_products(self, product_id, n=5):
        # Retrieve top-N similar products based on the combined similarity matrix
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        product_similarities = self.combined_similarity_matrix[product_idx]
        top_n_indices = np.argsort(product_similarities)[::-1][1:n+1]
        return self.products_df.iloc[top_n_indices][['category_name', 'product_id', 'product_name']]

    def multi_hop_recommendation(self, user_id, hop=2, top_n=5, exclude_purchased=False):
        # Generate multi-hop recommendations
        user_node = f"user_{user_id}"
        recommendations = {}

        # Get products the user has "purchased"
        purchased_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph[user_node][neighbor]['interaction'] == 'purchased'
        }

        # Get the direct neighbors of the user (first-hop)
        neighbors = set(self.graph.neighbors(user_node))
        print(f"First-hop neighbors for user '{user_id}': {neighbors}")  # Debugging: Print first-hop neighbors

        for neighbor in neighbors:
            if hop == 1:
                if self.graph.nodes[neighbor]['type'] == 'product':
                    recommendations[neighbor] = self.graph[user_node][neighbor]['weight']
            else:
                # Get second-hop neighbors (products connected to the user's first-hop products)
                second_hop_neighbors = set(self.graph.neighbors(neighbor))
                print(f"Second-hop neighbors for product '{neighbor}': {second_hop_neighbors}")  # Debugging: Print second-hop neighbors
                
                for second_neighbor in second_hop_neighbors:
                    if self.graph.nodes[second_neighbor]['type'] == 'user' and second_neighbor != user_node:
                        # Get products connected to this second-hop user
                        third_hop_products = set(self.graph.neighbors(second_neighbor))
                        for product in third_hop_products:
                            if self.graph.nodes[product]['type'] == 'product' and product not in neighbors:
                                # Add to recommendations, using addition of weights for simplicity
                                score = self.graph[user_node][neighbor]['weight'] + self.graph[neighbor][second_neighbor]['weight']
                                recommendations[product] = recommendations.get(product, 0) + score

        # If exclude_purchased is True, filter out products the user has already purchased
        if exclude_purchased:
            recommendations = {prod: score for prod, score in recommendations.items() if prod not in purchased_products}

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Extract category_name for each recommended product
        return [(self.products_df[self.products_df['product_id'] == int(item[0].split('_')[1])]['category_name'].values[0],
                 int(item[0].split('_')[1]),
                 item[1]) for item in sorted_recommendations]

    def run_recommendation_pipeline(self):
        # Complete pipeline to generate recommendations
        self.prepare_product_documents()
        self.generate_bert_embeddings()
        self.compute_similarity_matrix()
        self.construct_user_product_graph()
        self.compute_combined_similarity()


# Example Usage
if __name__ == "__main__":
    recommender = HybridRecommendationSystem('data/products.csv', 'data/transactions.csv')
    recommender.run_recommendation_pipeline()

    # Example: Get recommendations for user 101 with two-hop traversal, limited to top 5, excluding already purchased products
    customer_id = 101
    print(f"Top 5 recommendations for user '{customer_id}' with two-hop traversal (excluding purchased):")
    user_recommendations = recommender.multi_hop_recommendation(customer_id, hop=2, top_n=5, exclude_purchased=True)

    # Print the formatted table using the imported function
    print_formatted_recommendations(user_recommendations, recommender.products_df)
