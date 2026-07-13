import os
import argparse
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Import our utility functions
import utils

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train_model(epochs=20, batch_size=32, quick=False):
    print("=================================================================")
    print("🍕 Starting Food Category Recognition Model Training Pipeline 🍕")
    print("=================================================================")
    
    # 1. Initialize assets (moves generated banner/logo if available)
    utils.init_assets()
    
    # Create graphs directory if not exists
    os.makedirs('graphs', exist_ok=True)
    
    # Define dataset folders
    train_dir = os.path.join('food11 dataset', 'training')
    val_dir = os.path.join('food11 dataset', 'validation')
    eval_dir = os.path.join('food11 dataset', 'evaluation')
    
    print("\n[Step 1] Loading files from dataset directory...")
    # Load dataframes using utilities
    try:
        train_df = utils.get_dataset_df(train_dir)
        val_df = utils.get_dataset_df(val_dir)
        eval_df = utils.get_dataset_df(eval_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check that the 'food11 dataset/' folder is correctly placed in the directory.")
        sys.exit(1)
        
    print(f"-> Found {len(train_df)} training images.")
    print(f"-> Found {len(val_df)} validation images.")
    print(f"-> Found {len(eval_df)} evaluation images.")
    
    if quick:
        print("\n⚡ [Quick Mode Enabled] Downsampling dataset for a rapid sanity test...")
        train_df = train_df.sample(min(len(train_df), 150), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(min(len(val_df), 50), random_state=42).reset_index(drop=True)
        eval_df = eval_df.sample(min(len(eval_df), 50), random_state=42).reset_index(drop=True)
        epochs = 2
        print(f"-> New counts - Train: {len(train_df)}, Val: {len(val_df)}, Eval: {len(eval_df)}. Epochs set to {epochs}.")
        
    # Get classes in exact index order 0 to 10
    classes = [utils.CLASS_MAP[i] for i in range(11)]
    print(f"-> Targeted classes (11 categories): {classes}")
    
    # 2. Image Generators with Augmentation
    print("\n[Step 2] Setting up ImageDataGenerators with augmentation...")
    # Training datagen with specified augmentations
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation and evaluation generators should ONLY rescale, no augmentation
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    eval_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Flow from dataframe
    target_size = (224, 224)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='filename',
        y_col='class_name',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_dir,
        x_col='filename',
        y_col='class_name',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )
    
    eval_generator = eval_datagen.flow_from_dataframe(
        dataframe=eval_df,
        directory=eval_dir,
        x_col='filename',
        y_col='class_name',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )
    
    # 3. Model Architecture (MobileNetV2 Transfer Learning)
    print("\n[Step 3] Constructing model architecture (MobileNetV2 base)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base MobileNetV2 model
    base_model.trainable = False
    print("-> Frozen MobileNetV2 base layers.")
    
    # Custom Dense Head for Food Classification
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(11, activation='softmax')
    ])
    
    model.summary()
    
    # 4. Compilation
    print("\n[Step 4] Compiling model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. Callbacks
    print("\n[Step 5] Configuring Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath='food_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # 6. Model Training
    print(f"\n[Step 6] Running model fit for {epochs} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print("\n[Step 7] Model training completed. Evaluating results...")
    
    # 7. Model Evaluation
    # Load the best model saved during training (in case EarlyStopping rolled back or model checkpoint saved a better one)
    if os.path.exists('food_model.keras'):
        print("-> Loading best checkpointed model for final evaluation...")
        model = tf.keras.models.load_model('food_model.keras')
        
    # Evaluate accuracy metrics
    train_loss, train_acc = model.evaluate(train_generator, verbose=0)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    eval_loss, eval_acc = model.evaluate(eval_generator, verbose=0)
    
    print(f"\n=== FINAL PERFORMANCE METRICS ===")
    print(f"Training Accuracy:   {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Evaluation Accuracy: {eval_acc*100:.2f}%")
    print(f"=================================")
    
    # Get True labels and predicted labels for evaluation set
    print("\nGenerating Classification Report and Confusion Matrix...")
    eval_generator.reset()
    predictions = model.predict(eval_generator, steps=len(eval_generator), verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = eval_generator.classes
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    # Save the classification report text to a file for stream-lit application to render
    with open('graphs/classification_report.txt', 'w') as f:
        f.write(report)
        
    # Plot curves and matrix using utilities
    utils.plot_history(history, 'graphs')
    print("-> Saved accuracy and loss graphs to 'graphs/' directory.")
    
    utils.plot_confusion_matrix_heatmap(y_true, y_pred, classes, 'graphs')
    print("-> Saved confusion matrix plot to 'graphs/' directory.")
    
    # 8. Serialization of files
    print("\n[Step 8] Saving variables to pickle files...")
    utils.save_pickle(classes, 'class_names.pkl')
    utils.save_pickle(history.history, 'history.pkl')
    
    accuracy_metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'eval_acc': eval_acc
    }
    utils.save_pickle(accuracy_metrics, 'accuracy.pkl')
    
    print("-> Serialized: 'class_names.pkl', 'history.pkl', 'accuracy.pkl'")
    print("\n🎉 Training Pipeline Finished Successfully! 🎉\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Food Category Recognition CNN Training Script")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--quick', action='store_true', help="Run a quick 2-epoch sanity training with downsampled data")
    
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, quick=args.quick)
