"""
Polynomial regression with OLS and Ridge regularization.

This module implements polynomial regression using both ordinary least squares
(OLS) and Ridge regularization to handle overfitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict


class PolynomialRegression:
    """
    Polynomial regression with OLS and Ridge regularization.
    
    Fits polynomial models of specified degree to data, with optional
    Ridge (L2) regularization to prevent overfitting.
    
    Parameters
    ----------
    degree : int
        Degree of polynomial (default: 10)
    
    Attributes
    ----------
    degree : int
        Polynomial degree
    coef_ols_ : np.ndarray
        OLS coefficients (after fitting)
    coef_ridge_ : np.ndarray
        Ridge coefficients (after fitting)
    
    Examples
    --------
    >>> X = np.array([-9, 13, 14, -19, 16])
    >>> y = np.array([1.72e11, 4.78e11, 1.01e12, 2.71e13, 3.83e12])
    >>> model = PolynomialRegression(degree=10)
    >>> model.fit(X, y, lambda_ridge=1e4)
    >>> y_pred = model.predict(X, method='ridge')
    """
    
    def __init__(self, degree: int = 10):
        """Initialize polynomial regression model."""
        if degree < 1:
            raise ValueError("Polynomial degree must be at least 1")
        
        self.degree = degree
        self.coef_ols_ = None
        self.coef_ridge_ = None
        self._fitted = False
    
    def _create_vandermonde(self, x: np.ndarray) -> np.ndarray:
        """
        Create Vandermonde matrix for polynomial features.
        
        Parameters
        ----------
        x : np.ndarray
            Input data (1D array)
            
        Returns
        -------
        X : np.ndarray
            Vandermonde matrix of shape (n, degree+1)
        """
        return np.vander(x, N=self.degree + 1, increasing=True)
    
    def fit(self, x: np.ndarray, y: np.ndarray, lambda_ridge: float = 1e4) -> 'PolynomialRegression':
        """
        Fit polynomial regression using OLS and Ridge.
        
        Parameters
        ----------
        x : array_like
            Input features (1D array)
        y : array_like
            Target values (1D array)
        lambda_ridge : float, optional
            Ridge regularization parameter (default: 1e4)
            
        Returns
        -------
        self : PolynomialRegression
            Fitted model
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        # Create design matrix
        X = self._create_vandermonde(x)
        
        # OLS solution: alpha = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        
        try:
            self.coef_ols_ = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.coef_ols_ = np.full(self.degree + 1, np.nan)
        
        # Ridge solution: alpha = (X'X + λI)^{-1} X'y
        try:
            self.coef_ridge_ = np.linalg.solve(
                XtX + lambda_ridge * np.eye(self.degree + 1),
                Xty
            )
        except np.linalg.LinAlgError:
            self.coef_ridge_ = np.full(self.degree + 1, np.nan)
        
        self._fitted = True
        self.lambda_ridge_ = lambda_ridge
        
        return self
    
    def predict(self, x: np.ndarray, method: str = 'ridge') -> np.ndarray:
        """
        Make predictions using fitted model.
        
        Parameters
        ----------
        x : array_like
            Input features
        method : str, optional
            Prediction method: 'ols' or 'ridge' (default: 'ridge')
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        x = np.asarray(x, dtype=float)
        X = self._create_vandermonde(x)
        
        if method == 'ols':
            coef = self.coef_ols_
        elif method == 'ridge':
            coef = self.coef_ridge_
        else:
            raise ValueError("method must be 'ols' or 'ridge'")
        
        with np.errstate(invalid='ignore', over='ignore'):
            y_pred = X @ coef
        
        return y_pred
    
    def plot_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_range: Optional[Tuple[float, float]] = None,
        n_points: int = 500,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Plot the fitted polynomial curves.
        
        Parameters
        ----------
        x : np.ndarray
            Original x data
        y : np.ndarray
            Original y data
        x_range : tuple, optional
            (min, max) range for plotting. If None, uses data range ± 2
        n_points : int, optional
            Number of points for smooth curve (default: 500)
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save figure
        title : str, optional
            Custom plot title
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Determine plot range
        if x_range is None:
            x_range = (min(x) - 2, max(x) + 2)
        
        # Generate smooth curve
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y_ols = self.predict(x_grid, method='ols')
        y_ridge = self.predict(x_grid, method='ridge')
        
        # Filter out invalid values
        valid_ols = np.isfinite(y_ols)
        valid_ridge = np.isfinite(y_ridge)
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.scatter(x, y, color='black', s=50, zorder=3, label='Data points')
        
        if np.any(valid_ols):
            plt.plot(
                x_grid[valid_ols],
                y_ols[valid_ols],
                color='red',
                linewidth=2,
                label='OLS fit'
            )
        
        if np.any(valid_ridge):
            plt.plot(
                x_grid[valid_ridge],
                y_ridge[valid_ridge],
                color='blue',
                linewidth=2,
                label=f'Ridge fit (λ={self.lambda_ridge_:.0e})'
            )
        
        if title is None:
            title = f'Polynomial Regression (degree={self.degree}): OLS vs Ridge'
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compute_mse(self, x: np.ndarray, y: np.ndarray, method: str = 'ridge') -> float:
        """
        Compute mean squared error.
        
        Parameters
        ----------
        x : np.ndarray
            Input features
        y : np.ndarray
            True values
        method : str, optional
            Prediction method: 'ols' or 'ridge'
            
        Returns
        -------
        mse : float
            Mean squared error
        """
        y_pred = self.predict(x, method=method)
        valid = np.isfinite(y_pred)
        
        if not np.any(valid):
            return np.inf
        
        return float(np.mean((y[valid] - y_pred[valid])**2))
    
    def summary(self, x: np.ndarray, y: np.ndarray) -> str:
        """
        Generate model summary.
        
        Parameters
        ----------
        x : np.ndarray
            Input features
        y : np.ndarray
            Target values
            
        Returns
        -------
        summary : str
            Formatted summary string
        """
        if not self._fitted:
            return "Model not fitted yet"
        
        mse_ols = self.compute_mse(x, y, method='ols')
        mse_ridge = self.compute_mse(x, y, method='ridge')
        
        n_valid_ols = np.sum(np.isfinite(self.coef_ols_))
        n_valid_ridge = np.sum(np.isfinite(self.coef_ridge_))
        
        summary = f"""
Polynomial Regression Summary
==============================
Model Configuration:
  Degree: {self.degree}
  Number of coefficients: {self.degree + 1}

OLS Results:
  Valid coefficients: {n_valid_ols}/{self.degree + 1}
  MSE: {mse_ols:.2e}

Ridge Results:
  Lambda: {self.lambda_ridge_:.2e}
  Valid coefficients: {n_valid_ridge}/{self.degree + 1}
  MSE: {mse_ridge:.2e}

Model Selection:
  {"Ridge regression" if mse_ridge < mse_ols else "OLS"} has lower MSE
"""
        
        return summary

