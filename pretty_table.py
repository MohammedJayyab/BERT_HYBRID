import pandas as pd
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init
import json
import arabic_reshaper
from bidi.algorithm import get_display  # to display reshaped text correctly

# Initialize colorama for colored output
init(autoreset=True)

def fix_arabic_text(text):
    """
    Reshape Arabic text and apply the bidirectional algorithm to display it correctly.
    Truncate the text to a maximum of 50 characters and add "..." if it's cut.
    """
    reshaped_text = arabic_reshaper.reshape(text)  # Reshape the Arabic letters
    bidi_text = get_display(reshaped_text)  # Correct the direction for right-to-left display
    
    if len(bidi_text) > 50:
        return bidi_text[:50] + '...'  # Truncate the text and add "..." if cut
    return bidi_text

def print_formatted_recommendations(recommendations, products_df, output_file="output.txt"):
    # Convert the list of recommendations to a DataFrame
    recommendation_data = []

    for category_name, product_id, score, filter_type in recommendations:
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        recommendation_data.append({
            'category_id': product_info['category_id'],
            'category_name': fix_arabic_text(category_name),  # Fix Arabic text
            'product_id': product_id,
            'product_name_ar':fix_arabic_text( product_info['product_name_ar']),            
            'product_name': fix_arabic_text(product_info['product_name']),  # Fix Arabic text
            'score': f"{score:.4f}",
            'filter_type': filter_type
        })

    # Create a DataFrame from the recommendation data
    recommendations_df = pd.DataFrame(recommendation_data)

    # Sort by score (if not already sorted)
    recommendations_df = recommendations_df.sort_values(by='score', ascending=False)

    # Define color based on filter_type
    def get_colored_row(row):
        filter_type = row['filter_type']
        if filter_type == 'content-based':
            color = Fore.GREEN
        elif filter_type == 'collaborative':
            color = Fore.BLUE
        else:
            color = Fore.YELLOW

        # Apply the color to the entire row and return as a list
        return [color + str(item) + Style.RESET_ALL for item in row]

    # Apply color to each row based on filter_type
    colored_rows = recommendations_df.apply(get_colored_row, axis=1).values.tolist()

    # Format the table using tabulate and include headers
    table_string = tabulate(colored_rows, headers=recommendations_df.columns, tablefmt='fancy_grid', showindex=False)

    # Save the formatted table to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table_string)
        print(f"Table saved to {output_file}.")

    # Now print the same table to the console
    print(table_string)

def export_recommendations_as_json(recommendations, products_df, customer_id):
    """
    Export recommendations as JSON format for a specific customer.
    Ensure that any non-serializable types (like numpy int64) are converted to standard Python types.
    """
    recommendation_data = []
    
    for category_name, product_id, score, filter_type in recommendations:
        # Retrieve product information from products_df
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        recommendation_data.append({
            'category_id': int(product_info['category_id']),  # Convert to native Python int
            'category_name': (category_name),  # Fix Arabic text
            'product_id': str(product_id),  # Convert to string if needed
            'product_name': (product_info['product_name']),  # Fix Arabic text
            'product_name_ar': product_info['product_name_ar'],
            'score': round(float(score), 4),  # Convert to native Python float and limit to 4 decimal places
            'filter_type': filter_type
        })
    
    # Return JSON formatted recommendation data
    return {customer_id: recommendation_data}

def save_recommendations_to_json(file_path, recommendations_dict):
    """
    Save all customer recommendations to a JSON file.
    Ensure that any non-serializable types (like numpy int64) are converted to standard Python types.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # Convert any non-serializable numpy types to native Python types
        json.dump(recommendations_dict, f, ensure_ascii=False, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else float(o))
# method to save the recommendations to csv file
def save_recommendations_to_csv(file_path, recommendations_dict):
    """
    Save all customer recommendations to a CSV file.
    Ensure that any non-serializable types (like numpy int64) are converted to standard Python types.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # Convert any non-serializable numpy types to native Python types
        json.dump(recommendations_dict, f, ensure_ascii=False, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else float(o))
 
    
