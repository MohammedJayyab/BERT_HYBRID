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
        self.interaction_weights = interaction_weights if interaction_weights else {'clicked': 1, 'added': 2, 'purchased': 3}
        
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


    def generate_bert_embeddings__OLD(self):
        # Generate BERT embeddings for each product document
        print("1/2:: Generating BERT embeddings for product documents...")
        def get_bert_embedding(text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        print("2/2:: Generating BERT embeddings for product documents...")
        
        self.products_df['embedding'] = self.products_df['document'].apply(get_bert_embedding)
        print("Apply embedding to product document......[ok]")
        self.product_vectors = np.vstack(self.products_df['embedding'].values)

        print("BERT embeddings generated successfully......[ok]")
    
    def generate_bert_embeddings(self, batch_size=16):
        # Generate BERT embeddings for product documents in batches
        print("Generating BERT embeddings for product documents...")

        documents = self.products_df['document'].tolist()  # Get all documents as a list

        # If documents are empty, raise an error early
        if len(documents) == 0:
            print("Error: No documents found to generate embeddings.")
            return

        embeddings = []

        # Move model to GPU if available
        if torch.cuda.is_available():
            print("Using GPU for BERT model.")
            self.model.cuda()
        else:
            print("Using CPU for BERT model.")

        # Process documents in batches to improve performance
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Tokenize the batch and print debugging info
            print(f"Processing batch {i // batch_size + 1}/{len(documents) // batch_size + 1}. Number of documents in this batch: {len(batch)}")
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get the mean of the last hidden state and move back to CPU for NumPy processing
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Check if any embeddings are generated
            if batch_embeddings.size == 0:
                print(f"Error: No embeddings generated for batch {i // batch_size + 1}.")
                continue

            embeddings.append(batch_embeddings)
            print(f"Processed batch {i // batch_size + 1} successfully.")

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




    def construct_user_product_graph__old(self):
        # Add product nodes to the graph
        for _, row in self.products_df.iterrows():
            self.graph.add_node(f"product_{row['product_id']}", type='product')

        # Add user-product interactions to the graph
        for _, row in self.transactions_df.iterrows():
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type'] #if pd.notna(row['interaction_type']) else 'clicked'  # Default to 'clicked' if NaN
            quantity = row.get('quantity', 1)  # Default to 1 if quantity is not present
            base_weight = self.interaction_weights.get(interaction_type, 1)
            weight = base_weight * np.log(quantity + 1)  # Final weight is based on interaction type and quantity with logarithmic scaling

            if user_node not in self.graph:
                self.graph.add_node(user_node, type='user')
            
            # Add an edge with interaction type, weight, and quantity
            self.graph.add_edge(user_node, product_node, weight=weight, interaction=interaction_type, quantity=quantity)
    def construct_user_product_graph(self):
        # Add product nodes to the graph
        print("construct user product graph.......................")
        for _, row in self.products_df.iterrows():
            self.graph.add_node(f"product_{row['product_id']}", type='product')

        # Add user-product interactions to the graph
        for _, row in self.transactions_df.iterrows():
            user_node = f"user_{row['customer_id']}"
            product_node = f"product_{row['product_id']}"
            interaction_type = row['interaction_type']# if pd.notna(row['interaction_type']) else 'clicked'  # Default to 'clicked' if NaN
            quantity = row.get('quantity', 1)  # Default to 1 if quantity is not present
            base_weight = self.interaction_weights.get(interaction_type, 1)
            weight = base_weight * np.log(quantity + 1)  # Final weight is based on interaction type and quantity with logarithmic scaling

            # Ensure that both user and product nodes are assigned the correct type
            if user_node not in self.graph:
                self.graph.add_node(user_node, type='user')  # Add user node with 'user' type
            
            if product_node not in self.graph:
                self.graph.add_node(product_node, type='product')  # Add product node with 'product' type
            
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

    def recommend_new_or_non_interacted_products(self, user_id, n=5):
        # Recommend new products that the user has not interacted with
        user_node = f"user_{user_id}"
        
        if user_node not in self.graph:
            print(f"User '{user_id}' not found in the graph.")
            return []

        interacted_products = {
            neighbor for neighbor in self.graph.neighbors(user_node)
            if self.graph.nodes[neighbor]['type'] == 'product'
        }
        
        all_products = {f"product_{row['product_id']}" for _, row in self.products_df.iterrows()}
        non_interacted_products = all_products - interacted_products

        recommendations = []
        
        # Calculate the similarity only with products interacted by the user
        interacted_indices = [self.products_df[self.products_df['product_id'] == p.split('_')[1]].index[0] for p in interacted_products]

        for product in non_interacted_products:
            product_index = self.products_df[self.products_df['product_id'] == product.split('_')[1]].index[0]
            
            # Calculate similarity only with interacted products and take the maximum
            if interacted_indices:
                content_similarity = np.max(self.similarity_matrix[product_index, interacted_indices])
            else:
                content_similarity = 0

            # Only content similarity is used as there's no interaction or collaborative data for new products
            score = 1 * content_similarity  
            
            recommendations.append((self.products_df.iloc[product_index]['category_name'],
                                    self.products_df.iloc[product_index]['product_id'],
                                    self.products_df.iloc[product_index]['product_name'],
                                    score))

        # Sort recommendations by score in descending order and limit to top N
        sorted_recommendations = sorted(recommendations, key=lambda x: x[3], reverse=True)[:n]

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
                item[1]) for item in sorted_recommendations]

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
if __name__ == "__main__":
    recommender = HybridRecommendationSystem('data/ml_products.csv', 'data/ml_transactions.csv')
    recommender.run_recommendation_pipeline()

    # Example: Get top 5 similar products for product with ID
    product_id = "5010436603022"
    print(f"Top 5 similar products for product '{product_id}':")
    similar_products = recommender.get_top_n_similar_products(product_id, n=5)
    # Print the formatted table using the imported function
    print_formatted_recommendations(similar_products[['category_name', 'product_id', 'score']].values.tolist(), recommender.products_df)

    # Example: Get recommendations for user 101 with two-hop traversal, limited to top 5, excluding already purchased products
    customer_id = "9568245350259500"
    print(f"Top 5 recommendations for user '{customer_id}' with two-hop traversal (excluding purchased):")
    user_recommendations = recommender.multi_hop_recommendation(customer_id, hop=2, top_n=10, exclude_purchased=True)

    # Extract only the relevant three values: category_name, product_id, and score for `print_formatted_recommendations` function
    user_recommendations_trimmed = [(category_name, product_id, score) for category_name, product_id, _, score in user_recommendations]

    # Print the formatted table using the imported function
    print_formatted_recommendations(user_recommendations_trimmed, recommender.products_df)

    # Example: Get recommendations for new or non-interacted products for user 101
    print(f"Top 5 new or non-interacted product recommendations for user '{customer_id}':")
    new_recommendations = recommender.recommend_new_or_non_interacted_products(customer_id, n=10)

    # Extract only the relevant three values: category_name, product_id, and score for `print_formatted_recommendations` function
    new_recommendations_trimmed = [(category_name, product_id, score) for category_name, product_id, _, score in new_recommendations]

    # Print the formatted table using the imported function
    print_formatted_recommendations(new_recommendations_trimmed, recommender.products_df)

    # Example: Blend user recommendations with new recommendations
    print("***************************************************************")
    print(f"Top 10 blended recommendations for user '{customer_id}':")
    print("***************************************************************")
    # Assuming you have user_recommendations_trimmed and new_recommendations_trimmed defined
    blended_recommendations = recommender.blend_user_and_new_recommendations(
        user_recommendations_trimmed,
        new_recommendations_trimmed,
        user_ratio=0.5,   # 50% from user recommendations
        new_ratio=0.5,    # 50% from new recommendations        
    )

    # Print the blended recommendations using print_formatted_recommendations
    print_formatted_recommendations(blended_recommendations, recommender.products_df)
