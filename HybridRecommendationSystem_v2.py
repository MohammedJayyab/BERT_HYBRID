import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import the print_formatted_recommendations function from pretty_table.py
from pretty_table import print_formatted_recommendations

class HybridRecommendationSystem:
    def __init__(self, product_file, transaction_file, interaction_weights=None, blend_weights=None):
        # Load data
        self.products_df = pd.read_csv(product_file)
        self.transactions_df = pd.read_csv(transaction_file)
        self.graph = nx.Graph()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.product_vectors = None
        self.similarity_matrix = None
        self.combined_similarity_matrix = None
        self.min_max_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()

        # Customizable interaction weights for user behavior (clicked, added, purchased)
        self.interaction_weights = interaction_weights if interaction_weights else {'clicked': 1, 'added': 2, 'purchased': 3}
        
        # Hyperparameters to blend collaborative and content-based features
        self.blend_weights = blend_weights if blend_weights else {'content': 0.5, 'collaborative': 0.5}

        # Weight parameters for interaction weight, content similarity, and collaborative similarity
        self.w1 = 0.2  # Interaction Weight
        self.w2 = 0.5  # Content Similarity
        self.w3 = 0.3  # Collaborative Similarity

    def prepare_product_documents(self):
        # Concatenate product name, category, and description
        self.products_df['document'] = self.products_df['category_name'] + " : " + \
                                       self.products_df['product_name'] + " : " + \
                                       self.products_df['description']

    def generate_bert_embeddings(self):
        # Generate BERT embeddings for each product document
        def get_bert_embedding(text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        self.products_df['embedding'] = self.products_df['document'].apply(get_bert_embedding)
        self.product_vectors = np.vstack(self.products_df['embedding'].values)

    def compute_similarity_matrix(self):
        # Compute product-to-product cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.product_vectors)
        # Normalize the similarity matrix to be between 0 and 1 using Min-Max normalization
        self.similarity_matrix = self.min_max_scaler.fit_transform(self.similarity_matrix)

    def construct_user_product_graph(self):
        # Add product nodes to the graph
        for _, row in self.products_df.iterrows():
            self.graph.add_node(f"product_{row['product_id']}", type='product')

        # Add user-product interactions to the graph
        for _, row in self.transactions_df.iterrows():
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type'] if pd.notna(row['interaction_type']) else 'clicked'  # Default to 'clicked' if NaN
            quantity = row.get('quantity', 1)  # Default to 1 if quantity is not present
            base_weight = self.interaction_weights.get(interaction_type, 1)
            weight = base_weight * np.log(quantity + 1)  # Final weight is based on interaction type and quantity with logarithmic scaling

            if user_node not in self.graph:
                self.graph.add_node(user_node, type='user')
            
            # Add an edge with interaction type, weight, and quantity
            self.graph.add_edge(user_node, product_node, weight=weight, interaction=interaction_type, quantity=quantity)

    def normalize_edge_weights(self):
        # Normalize edge weights in the user-product graph to be between 0 and 1
        all_weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
        weights_array = np.array(all_weights).reshape(-1, 1)
        normalized_weights = self.min_max_scaler.fit_transform(weights_array).flatten()

        # Update graph with normalized weights
        for idx, (u, v, d) in enumerate(self.graph.edges(data=True)):
            self.graph[u][v]['weight'] = normalized_weights[idx]

    def compute_combined_similarity(self):
        # Combine collaborative similarity with content-based similarity
        num_products = len(self.products_df)
        self.combined_similarity_matrix = np.zeros((num_products, num_products))
        
        collaborative_similarities = []
        for i in range(num_products):
            for j in range(num_products):
                if i != j:
                    product_i = f"product_{self.products_df.iloc[i]['product_id']}"
                    product_j = f"product_{self.products_df.iloc[j]['product_id']}"

                    common_users = len(set(self.graph.neighbors(product_i)) & set(self.graph.neighbors(product_j)))
                    collaborative_similarity = common_users / (len(set(self.graph.neighbors(product_i))) * len(set(self.graph.neighbors(product_j))) + 1e-9)
                    collaborative_similarities.append(collaborative_similarity)

        # Normalize collaborative similarities using Z-score normalization
        collaborative_similarities = np.array(collaborative_similarities).reshape(-1, 1)
        normalized_collaborative_similarities = self.standard_scaler.fit_transform(collaborative_similarities).flatten()
        
        idx = 0
        for i in range(num_products):
            for j in range(num_products):
                if i != j:
                    content_similarity = self.similarity_matrix[i, j]
                    collaborative_similarity = normalized_collaborative_similarities[idx]
                    idx += 1
                    # Combine normalized collaborative similarity and content similarity using blend weights
                    self.combined_similarity_matrix[i, j] = (self.blend_weights['collaborative'] * collaborative_similarity +
                                                             self.blend_weights['content'] * content_similarity)

    def get_top_n_similar_products(self, product_id, n=5):
        # Retrieve top-N similar products based on the combined similarity matrix
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        product_similarities = self.combined_similarity_matrix[product_idx]
        top_n_indices = np.argsort(product_similarities)[::-1][1:n+1]

        # Create a DataFrame for the top N products with similarity scores
        similar_products = self.products_df.iloc[top_n_indices][['category_name', 'product_id', 'product_name']].copy()
        similar_products['score'] = product_similarities[top_n_indices]

        return similar_products

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
                    product_index = self.products_df[self.products_df['product_id'] == int(neighbor.split('_')[1])].index[0]
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
                                product_index = self.products_df[self.products_df['product_id'] == int(product.split('_')[1])].index[0]
                                content_similarity = self.similarity_matrix[product_index, product_index]
                                collaborative_similarity = self.combined_similarity_matrix[product_index, product_index]
                                
                                # Final scoring function incorporating interaction weight, content similarity, and collaborative similarity
                                score = (self.w1 * interaction_weight +
                                         self.w2 * content_similarity +
                                         self.w3 * collaborative_similarity)                                
                                recommendations[product] = recommendations.get(product, 0) + score
                                
                                
                                
                                #print(f"Product '{product}' added with score: {score_total}")  # Debugging: Print product and score
                                # if(product == 'product_6'):
                                #     #print(f"tt: for product ID 6:-> {tt}")
                                #     #print(f"Score for product ID 6: {score}")
                                #     print(f"self.w1 * interaction_weight : {self.w1 * interaction_weight }")
                                #     print(f"self.w2 * content_similarity : {self.w2 * content_similarity }")
                                #     print(f"self.w3 * collaborative_similarity : {self.w3 * collaborative_similarity }")
                                

        # If exclude_purchased is True, filter out products the user has already purchased
        if exclude_purchased:
            recommendations = {prod: score for prod, score in recommendations.items() if prod not in purchased_products}

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # Assuming sorted_recommendations is the result list
    

            
        # Extract category_name, product_id, product_name, and score for each recommended product
        return [(self.products_df[self.products_df['product_id'] == int(item[0].split('_')[1])]['category_name'].values[0],
                 int(item[0].split('_')[1]),
                 item[1]) for item in sorted_recommendations]

    def run_recommendation_pipeline(self):
        # Complete pipeline to generate recommendations
        self.prepare_product_documents()
        self.generate_bert_embeddings()
        self.compute_similarity_matrix()
        self.construct_user_product_graph()
        self.normalize_edge_weights()
        self.compute_combined_similarity()

# Example Usage
if __name__ == "__main__":
    recommender = HybridRecommendationSystem('data/products.csv', 'data/transactions.csv')
    recommender.run_recommendation_pipeline()

    # Example: Get top 5 similar products for product with ID
    product_id = 1
    print(f"Top 5 similar products for product '{product_id}':")
    similar_products = recommender.get_top_n_similar_products(product_id, n=5)
    # Print the formatted table using the imported function
    print_formatted_recommendations(similar_products[['category_name', 'product_id', 'score']].values.tolist(), recommender.products_df)

    # Example: Get recommendations for user 101 with two-hop traversal, limited to top 5, excluding already purchased products
    customer_id = 101
    print(f"Top 5 recommendations for user '{customer_id}' with two-hop traversal (excluding purchased):")
    user_recommendations = recommender.multi_hop_recommendation(customer_id, hop=2, top_n=5, exclude_purchased=True)

    # Extract only the relevant three values: category_name, product_id, and score
    user_recommendations_trimmed = [(category_name, product_id, score) for category_name, product_id, score in user_recommendations]

    # Print the formatted table using the imported function
    print_formatted_recommendations(user_recommendations_trimmed, recommender.products_df)
