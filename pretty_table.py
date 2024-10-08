import pandas as pd

def print_formatted_recommendations(recommendations, products_df):
    # Convert the list of recommendations to a DataFrame
    recommendation_data = []
    
    for category_name, product_id, score in recommendations:
        # Retrieve product information from products_df
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        recommendation_data.append({
            'category_id': product_info['category_id'],
            'category_name': category_name,
            'product_id': product_id,
            'product_name': product_info['product_name'],
            'score': score
        })
    
    # Create a DataFrame from the recommendation data
    recommendations_df = pd.DataFrame(recommendation_data)

    # Sort by score (if not already sorted)
    recommendations_df = recommendations_df.sort_values(by='score', ascending=False)

    # Print the DataFrame as a pretty table
    print(recommendations_df.to_string(index=False))


