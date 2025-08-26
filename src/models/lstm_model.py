"""
LSTM model implementation for the roller mill vibration prediction project.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

from .base_model import DeepLearningModel


class LSTMModel(DeepLearningModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, name: str = "LSTM", **kwargs):
        """Initialize the LSTM model.
        
        Args:
            name: Name of the model.
            **kwargs: Model parameters.
        """
        super().__init__(name, **kwargs)
        
        # LSTM-specific parameters
        self.sequence_length = kwargs.get('sequence_length', 60)
        self.units = kwargs.get('units', 50)
        self.layers_count = kwargs.get('layers', 2)
        self.dropout_rate = kwargs.get('dropout', 0.2)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.validation_split = kwargs.get('validation_split', 0.2)
        
        # Store model architecture parameters
        self.init_params.update({
            'sequence_length': self.sequence_length,
            'units': self.units,
            'layers': self.layers_count,
            'dropout': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience
        })
        
        self.scaler_X = None
        self.scaler_y = None
        
    def _build_model(self, input_shape: Tuple[int, ...], **kwargs) -> keras.Model:
        """Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features).
            **kwargs: Additional model parameters.
            
        Returns:
            Compiled Keras model.
        """
        model = keras.Sequential(name=f"{self.name}_model")
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        for i in range(self.layers_count):
            return_sequences = (i < self.layers_count - 1)  # Return sequences for all but last layer
            
            model.add(layers.LSTM(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_{i+1}'
            ))
            
            # Add batch normalization
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Dense layers for output
        model.add(layers.Dense(self.units // 2, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_final'))
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _prepare_lstm_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for LSTM training/prediction.
        
        Args:
            X: Input features.
            y: Target values (optional for prediction).
            
        Returns:
            Tuple of (X_sequences, y_sequences).
        """
        # Normalize features
        if self.scaler_X is None:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = self.scaler_X.transform(X)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            
            if y is not None:
                y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            
            # Normalize target if not already done
            if self.scaler_y is None:
                self.scaler_y = StandardScaler()
                y_sequences = self.scaler_y.fit_transform(y_sequences.reshape(-1, 1)).flatten()
            else:
                y_sequences = self.scaler_y.transform(y_sequences.reshape(-1, 1)).flatten()
        
        return X_sequences, y_sequences if y is not None else None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LSTMModel':
        """Fit the LSTM model to training data.
        
        Args:
            X: Training features.
            y: Training target.
            **kwargs: Additional fitting parameters.
            
        Returns:
            Self for method chaining.
        """
        self.logger.info(f"Training LSTM model with {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare LSTM data
        X_sequences, y_sequences = self._prepare_lstm_data(X, y)
        
        if len(X_sequences) < self.sequence_length:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Build model if not already built
        if self.model is None:
            input_shape = (self.sequence_length, X.shape[1])
            self.model = self._build_model(input_shape)
            
            self.logger.info(f"Built LSTM model with input shape {input_shape}")
            self.logger.info(f"Model summary:\n{self.model.summary()}")
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Train the model
        self.history = self.model.fit(
            X_sequences,
            y_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
        
        self.is_fitted = True
        
        # Store training information
        self.training_history.update({
            'training_samples': len(X_sequences),
            'features': X.shape[1],
            'sequence_length': self.sequence_length,
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None
        })
        
        self.logger.info(f"LSTM training completed. Final loss: {self.training_history['final_loss']:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the LSTM model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        # Prepare LSTM data
        X_sequences, _ = self._prepare_lstm_data(X)
        
        if len(X_sequences) == 0:
            self.logger.warning("No sequences available for prediction")
            return np.array([])
        
        # Make predictions
        predictions_scaled = self.model.predict(X_sequences, batch_size=self.batch_size, verbose=0)
        
        # Denormalize predictions
        if self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()
        else:
            predictions = predictions_scaled.flatten()
        
        # Extend predictions to match input length
        # First 'sequence_length' predictions are not available due to sequence requirement
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions
        
        return full_predictions
    
    def predict_sequence(self, X: pd.DataFrame, n_steps: int = 1) -> np.ndarray:
        """Predict multiple steps ahead using recursive prediction.
        
        Args:
            X: Initial features.
            n_steps: Number of steps to predict ahead.
            
        Returns:
            Multi-step predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(X) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for sequence prediction")
        
        # Prepare initial sequence
        X_scaled = self.scaler_X.transform(X)
        current_sequence = X_scaled[-self.sequence_length:]
        
        predictions = []
        
        for _ in range(n_steps):
            # Reshape for model input
            sequence_input = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict next step
            next_pred_scaled = self.model.predict(sequence_input, verbose=0)
            next_pred = self.scaler_y.inverse_transform(next_pred_scaled).flatten()[0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            # This is a simplified approach - in practice, you might want to use
            # the predicted value to update relevant features in the sequence
            current_sequence = current_sequence[1:]  # Remove oldest
            
            # For simplicity, we'll just append the last feature values
            # In a real scenario, you'd need to properly incorporate the prediction
            new_features = X_scaled[-1:].copy()  # Copy last feature row
            current_sequence = np.vstack([current_sequence, new_features])
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance for LSTM model using attention-like mechanism.
        
        Note: This is a simplified approximation of feature importance for LSTM.
        True importance would require more sophisticated methods like SHAP.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted or self.feature_names is None:
            return None
        
        # This is a placeholder implementation
        # For actual feature importance in LSTM, consider using:
        # - SHAP (SHapley Additive exPlanations)
        # - LIME (Local Interpretable Model-agnostic Explanations)
        # - Attention mechanisms
        # - Permutation importance
        
        # Simple approach: use the magnitude of input weights as proxy
        try:
            # Get the first LSTM layer
            first_lstm_layer = None
            for layer in self.model.layers:
                if isinstance(layer, layers.LSTM):
                    first_lstm_layer = layer
                    break
            
            if first_lstm_layer is not None:
                # Get input weights (simplified)
                weights = first_lstm_layer.get_weights()[0]  # Input weights
                
                # Calculate average absolute weight per feature
                if len(weights.shape) >= 2:
                    feature_weights = np.mean(np.abs(weights), axis=1)
                    
                    # Normalize to sum to 1
                    feature_weights = feature_weights / np.sum(feature_weights)
                    
                    return dict(zip(self.feature_names, feature_weights))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")
        
        return None
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history_dict = self.history.history
            
            plt.figure(figsize=(12, 4))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history_dict['loss'], label='Training Loss')
            if 'val_loss' in history_dict:
                plt.plot(history_dict['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(history_dict['mae'], label='Training MAE')
            if 'val_mae' in history_dict:
                plt.plot(history_dict['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")


class GRUModel(LSTMModel):
    """GRU model for time series prediction (similar to LSTM but with GRU layers)."""
    
    def __init__(self, name: str = "GRU", **kwargs):
        """Initialize the GRU model.
        
        Args:
            name: Name of the model.
            **kwargs: Model parameters.
        """
        super().__init__(name, **kwargs)
    
    def _build_model(self, input_shape: Tuple[int, ...], **kwargs) -> keras.Model:
        """Build the GRU model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features).
            **kwargs: Additional model parameters.
            
        Returns:
            Compiled Keras model.
        """
        model = keras.Sequential(name=f"{self.name}_model")
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # GRU layers
        for i in range(self.layers_count):
            return_sequences = (i < self.layers_count - 1)
            
            model.add(layers.GRU(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'gru_{i+1}'
            ))
            
            # Add batch normalization
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Dense layers for output
        model.add(layers.Dense(self.units // 2, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_final'))
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model


class ConvLSTMModel(LSTMModel):
    """Convolutional LSTM model for time series prediction."""
    
    def __init__(self, name: str = "ConvLSTM", **kwargs):
        """Initialize the ConvLSTM model.
        
        Args:
            name: Name of the model.
            **kwargs: Model parameters.
        """
        super().__init__(name, **kwargs)
        
        # ConvLSTM specific parameters
        self.conv_filters = kwargs.get('conv_filters', 32)
        self.kernel_size = kwargs.get('kernel_size', 3)
    
    def _build_model(self, input_shape: Tuple[int, ...], **kwargs) -> keras.Model:
        """Build the ConvLSTM model architecture.
        
        Args:
            input_shape: Shape of input data.
            **kwargs: Additional model parameters.
            
        Returns:
            Compiled Keras model.
        """
        model = keras.Sequential(name=f"{self.name}_model")
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Add a Conv1D layer before LSTM
        model.add(layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same',
            name='conv1d'
        ))
        
        # LSTM layers
        for i in range(self.layers_count):
            return_sequences = (i < self.layers_count - 1)
            
            model.add(layers.LSTM(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_{i+1}'
            ))
            
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Dense output layers
        model.add(layers.Dense(self.units // 2, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_final'))
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model