import os
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, classification_report

# 11 food classes mapping
CLASS_MAP = {
    0: "Bread",
    1: "Dairy Product",
    2: "Dessert",
    3: "Egg",
    4: "Fried Food",
    5: "Meat",
    6: "Noodles/Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable/Fruit"
}

# Funny messages for each class
FUNNY_MESSAGES = {
    "Bread": "Fresh from the bakery! 🥖",
    "Dairy Product": "Calcium Power! 🥛",
    "Dessert": "Too sweet to resist! 🍰",
    "Egg": "Egg-cellent choice! 🥚",
    "Fried Food": "Crispy happiness! 🍟",
    "Meat": "Protein loaded! 🥩",
    "Noodles/Pasta": "Slurp Mode Activated! 🍜",
    "Rice": "Every meal's best friend! 🍚",
    "Seafood": "Straight from the ocean! 🦐",
    "Soup": "Warm hugs in a bowl! 🥣",
    "Vegetable/Fruit": "Healthy eating unlocked! 🥗"
}

# Estimated calories and nutritional tip details
NUTRITIONAL_DATA = {
    "Bread": {
        "calories": "~265 kcal per 100g",
        "tip": "Opt for whole grain or sourdough breads to get more fiber, micronutrients, and a lower glycemic index!"
    },
    "Dairy Product": {
        "calories": "~60 - 350 kcal per 100g",
        "tip": "Dairy is a powerhouse of calcium and Vitamin D. Choose Greek yogurt or cottage cheese for high-quality protein."
    },
    "Dessert": {
        "calories": "~350 - 500 kcal per 100g",
        "tip": "Perfect for a sweet indulgence! Enjoy dessert mindfully as a treat, and try pairing it with berries to add fiber."
    },
    "Egg": {
        "calories": "~155 kcal per 100g (approx. 2 large eggs)",
        "tip": "Eggs are a gold standard for protein. The yolk contains essential choline and vitamins, so eat the whole egg!"
    },
    "Fried Food": {
        "calories": "~290 - 450 kcal per 100g",
        "tip": "Crispy and delicious, but high in saturated fats. Consider air-frying at home to get the crunch with 80% less oil!"
    },
    "Meat": {
        "calories": "~140 - 300 kcal per 100g",
        "tip": "Lean meats are fantastic for muscle building and recovery. Combine with green veggies to aid digestion."
    },
    "Noodles/Pasta": {
        "calories": "~130 kcal per 100g (cooked)",
        "tip": "Try whole wheat, chickpea, or edamame pasta to increase protein and fiber. Load it up with home-cooked tomato sauce!"
    },
    "Rice": {
        "calories": "~130 kcal per 100g (cooked)",
        "tip": "Rice is an excellent energy source. Brown, red, or wild rice retains the nutrient-rich outer bran layers."
    },
    "Seafood": {
        "calories": "~100 - 220 kcal per 100g",
        "tip": "Fish like salmon and mackerel are rich in Omega-3 fatty acids. Aim for seafood twice a week for heart and brain health."
    },
    "Soup": {
        "calories": "~50 - 150 kcal per 100g",
        "tip": "Vegetable or broth-based soups are highly hydrating, filling, and low in calories. A perfect healthy starter!"
    },
    "Vegetable/Fruit": {
        "calories": "~30 - 80 kcal per 100g",
        "tip": "Eat the rainbow! Different colors represent different antioxidants and vitamins that boost your immune system."
    }
}

# Food trivia facts
FOOD_FACTS = [
    "Honey is the only food that will never spoil. It can remain edible for thousands of years!",
    "Apples float in water because they are 25% air by volume.",
    "Bananas are botanically classified as berries, whereas strawberries are not!",
    "Pistachios are actually seeds of a fruit, not true botanical nuts.",
    "Broccoli contains more protein than steak per calorie, plus a wealth of vitamins and minerals.",
    "Carrots were originally purple or yellow. The orange variety was cultivated in the 17th century.",
    "Chocolate was used as currency by the Aztecs and Mayans in ancient Mesoamerica.",
    "Watermelon is 92% water, making it one of the most hydrating fruits on Earth.",
    "The first food eaten in space was applesauce, consumed by astronaut John Glenn in 1962.",
    "Pineapples take almost three years to grow and mature from a single seed!"
]

def get_random_food_fact():
    """Returns a random interesting food fact."""
    return random.choice(FOOD_FACTS)

def get_dataset_df(dir_path):
    """
    Scans the directory for images of the form classIndex_imageIndex.jpg,
    extracts the class index, maps it to class name, and returns a DataFrame.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' does not exist.")
    
    records = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # Filename structure is: <class_index>_<image_index>.jpg
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    class_idx = int(parts[0])
                    if class_idx in CLASS_MAP:
                        records.append({
                            'filename': filename,
                            'class_name': CLASS_MAP[class_idx]
                        })
                except ValueError:
                    # Ignore files that do not start with a valid integer class index
                    continue
                    
    df = pd.DataFrame(records)
    if len(df) == 0:
        print(f"Warning: No valid food images found in {dir_path}")
    return df

def save_pickle(data, filepath):
    """Saves data to a pickle file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_history(history, save_dir):
    """
    Plots training vs validation accuracy and loss graphs,
    and saves them to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if history is a dictionary or Keras History object
    h = history.history if hasattr(history, 'history') else history
    
    epochs = range(1, len(h['accuracy']) + 1)
    
    # 1. Accuracy Graph
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.plot(epochs, h['accuracy'], 'o-', color='#b388ff', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in h:
        plt.plot(epochs, h['val_accuracy'], 's-', color='#ff80ab', label='Validation Accuracy', linewidth=2)
    plt.title('Training & Validation Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', frameon=True, facecolor='#202020')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=300)
    plt.close()
    
    # 2. Loss Graph
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.plot(epochs, h['loss'], 'o-', color='#80d8ff', label='Training Loss', linewidth=2)
    if 'val_loss' in h:
        plt.plot(epochs, h['val_loss'], 's-', color='#ffd740', label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss Over Epochs', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='#202020')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_dir):
    """
    Computes a classification report and beautiful confusion matrix heatmap
    and saves the confusion matrix plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    plt.style.use('dark_background')
    
    # Create seaborn heatmap with customized color palette
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Purples', 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10}
    )
    
    plt.title('Confusion Matrix on Evaluation Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Class Label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def init_assets():
    """
    Creates default placeholder banner and logo images if they don't already exist.
    Safe to call on any deployment environment (local, Render, etc.).
    """
    assets_dir = 'assets'
    os.makedirs(assets_dir, exist_ok=True)
    
    banner_dest = os.path.join(assets_dir, 'banner.png')
    logo_dest = os.path.join(assets_dir, 'logo.png')
    
    # 1. Handle Banner
    if not os.path.exists(banner_dest):
        # Create a beautiful gradient placeholder banner
        img = Image.new('RGB', (1200, 400), color='#1e1b4b')
        draw = ImageDraw.Draw(img)
        # Add simple modern gradient look
        for y in range(400):
            r = int(0x1e + (0x7e - 0x1e) * (y / 400))
            g = int(0x1b + (0x22 - 0x1b) * (y / 400))
            b = int(0x4b + (0x9a - 0x4b) * (y / 400))
            draw.line([(0, y), (1200, y)], fill=(r, g, b))
        
        # Add some abstract graphic elements
        draw.ellipse([800, -100, 1300, 400], fill=None, outline='#ec4899', width=3)
        draw.ellipse([750, -150, 1350, 450], fill=None, outline='#8b5cf6', width=2)
        
        # Draw some text
        draw.text((100, 160), "🍕 Food Category Recognition", fill="#ffffff")
        draw.text((100, 220), "TensorFlow + MobileNetV2 + Streamlit Pipeline", fill="#a78bfa")
        img.save(banner_dest)
        print(f"Created default placeholder banner at {banner_dest}")

    # 2. Handle Logo
    if not os.path.exists(logo_dest):
        # Create a beautiful logo placeholder
        img = Image.new('RGB', (200, 200), color='#0f172a')
        draw = ImageDraw.Draw(img)
        # Draw gradient circle
        draw.ellipse([10, 10, 190, 190], fill='#8b5cf6')
        draw.ellipse([30, 30, 170, 170], fill='#0f172a')
        # Simple food silhouette placeholder
        draw.polygon([(100, 50), (60, 130), (140, 130)], fill='#f97316')
        draw.ellipse([80, 90, 95, 105], fill='#ef4444')
        draw.ellipse([110, 105, 120, 115], fill='#ef4444')
        img.save(logo_dest)
        print(f"Created default placeholder logo at {logo_dest}")

