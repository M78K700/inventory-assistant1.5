import openai
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from database import add_product, update_inventory_quantity, get_user_inventory

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

# Set the API key directly
openai.api_key = api_key

def get_inventory_context(inventory_df):
    """Convert inventory data to a context string for the chatbot"""
    if inventory_df.empty:
        return "The inventory is currently empty."
    
    context = "Current inventory status:\n"
    for category in inventory_df['category'].unique():
        category_items = inventory_df[inventory_df['category'] == category]
        context += f"\n{category}:\n"
        for _, item in category_items.iterrows():
            context += f"- {item['product_name']}: {item['quantity']} {item['unit']} (Min: {item['min_stock_level']})\n"
    
    return context

def process_inventory_command(user_id, command, inventory_df):
    """Process inventory-related commands"""
    try:
        # Convert command to lowercase for easier processing
        command = command.lower()
        
        # Check for add command
        if command.startswith('add'):
            parts = command.split()
            if len(parts) >= 3:
                try:
                    quantity = float(parts[1])
                    product_name = ' '.join(parts[2:])
                    
                    # Find matching product
                    matching_products = inventory_df[
                        inventory_df['product_name'].str.lower() == product_name.lower()
                    ]
                    
                    if not matching_products.empty:
                        product = matching_products.iloc[0]
                        add_product(user_id, product['product_name'], product['category'], 
                                  quantity, product['unit'], product['image_path'], 
                                  product['min_stock_level'])
                        return f"Added {quantity} {product['unit']} of {product['product_name']} to inventory."
                    else:
                        return f"Product '{product_name}' not found. Please add it through the 'Add Product' page first."
                except ValueError:
                    return "Invalid quantity. Please use format: 'add [quantity] [product name]'"
        
        # Check for use/remove command
        elif command.startswith(('use', 'remove')):
            parts = command.split()
            if len(parts) >= 3:
                try:
                    quantity = float(parts[1])
                    product_name = ' '.join(parts[2:])
                    
                    # Find matching product
                    matching_products = inventory_df[
                        inventory_df['product_name'].str.lower() == product_name.lower()
                    ]
                    
                    if not matching_products.empty:
                        product = matching_products.iloc[0]
                        if product['quantity'] >= quantity:
                            update_inventory_quantity(user_id, product['product_name'], quantity)
                            return f"Removed {quantity} {product['unit']} of {product['product_name']} from inventory."
                        else:
                            return f"Not enough {product['product_name']} in inventory. Current quantity: {product['quantity']} {product['unit']}"
                    else:
                        return f"Product '{product_name}' not found in inventory."
                except ValueError:
                    return "Invalid quantity. Please use format: 'use/remove [quantity] [product name]'"
        
        # Check for status command
        elif command.startswith('status'):
            parts = command.split()
            if len(parts) == 1:
                return get_inventory_context(inventory_df)
            elif len(parts) == 2:
                product_name = parts[1]
                matching_products = inventory_df[
                    inventory_df['product_name'].str.lower() == product_name.lower()
                ]
                if not matching_products.empty:
                    product = matching_products.iloc[0]
                    return f"Status of {product['product_name']}:\nQuantity: {product['quantity']} {product['unit']}\nMinimum Stock Level: {product['min_stock_level']}"
                else:
                    return f"Product '{product_name}' not found in inventory."
        
        return None
    except Exception as e:
        return f"Error processing command: {str(e)}"

def get_chatbot_response(user_id, user_input, inventory_df):
    """Get response from OpenAI chatbot based on user input and inventory context"""
    # First try to process as a command
    command_response = process_inventory_command(user_id, user_input, inventory_df)
    if command_response:
        return command_response
    
    # If not a command, use OpenAI for general conversation
    context = get_inventory_context(inventory_df)
    
    messages = [
        {"role": "system", "content": """You are an inventory management assistant. 
        You help users manage their grocery and household inventory. 
        You can answer questions about current inventory levels, suggest when to restock items,
        provide inventory management advice, and help with inventory optimization.
        
        Available commands:
        - add [quantity] [product name] - Add items to inventory
        - use/remove [quantity] [product name] - Remove items from inventory
        - status [product name] - Check status of a specific product
        - status - Check status of all inventory
        
        You can also ask general questions about inventory management."""},
        {"role": "system", "content": context},
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}" 