"""
Portfolio management module.
Handles portfolio CRUD operations and data persistence.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config.settings import get_settings


@dataclass
class Portfolio:
    """
    Represents an investment portfolio.
    """
    name: str
    tickers: List[str]
    weights: Dict[str, float]  # Ticker -> Weight (percentage, 0-100)
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate and normalize portfolio data."""
        # Ensure weights only contain tickers in the list
        self.weights = {t: self.weights.get(t, 0.0) for t in self.tickers}
    
    @property
    def total_weight(self) -> float:
        """Get total weight sum."""
        return sum(self.weights.values())
    
    @property
    def normalized_weights(self) -> Dict[str, float]:
        """Get normalized weights (sum to 1.0)."""
        total = self.total_weight
        if total == 0:
            return {t: 0.0 for t in self.tickers}
        return {t: w / total for t, w in self.weights.items()}
    
    def is_valid(self) -> bool:
        """Check if portfolio has valid configuration."""
        return len(self.tickers) > 0 and self.total_weight > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'tickers': self.tickers,
            'weights': self.weights,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Portfolio':
        """Create Portfolio from dictionary."""
        return cls(
            name=data.get('name', 'Unnamed'),
            tickers=data.get('tickers', []),
            weights=data.get('weights', {}),
            description=data.get('description', ''),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
        )
    
    @classmethod
    def from_legacy_format(cls, name: str, data: dict) -> 'Portfolio':
        """
        Create Portfolio from legacy portfolios.json format.
        Legacy format: {"tickers": [...], "weights": {...}}
        """
        return cls(
            name=name,
            tickers=data.get('tickers', []),
            weights=data.get('weights', {}),
            description=f"Imported from legacy portfolio",
        )


class PortfolioManager:
    """
    Manages portfolio storage and operations.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize portfolio manager.
        
        Args:
            storage_path: Path to portfolios JSON file
        """
        settings = get_settings()
        self.storage_path = storage_path or settings.portfolios_file
        self.legacy_path = settings.legacy_portfolios_file
        
        # Ensure data directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded portfolios
        self._portfolios: Dict[str, Portfolio] = {}
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure portfolios are loaded from disk."""
        if not self._loaded:
            self.load()
    
    def load(self) -> Dict[str, Portfolio]:
        """
        Load portfolios from storage file.
        
        Returns:
            Dictionary of portfolios
        """
        self._portfolios = {}
        
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, pdata in data.items():
                    self._portfolios[name] = Portfolio.from_dict(pdata)
                    
            except Exception as e:
                print(f"Error loading portfolios: {e}")
        
        self._loaded = True
        return self._portfolios
    
    def save(self) -> bool:
        """
        Save all portfolios to storage file.
        
        Returns:
            True if successful
        """
        try:
            data = {name: p.to_dict() for name, p in self._portfolios.items()}
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving portfolios: {e}")
            return False
    
    def get_all(self) -> Dict[str, Portfolio]:
        """Get all portfolios."""
        self._ensure_loaded()
        return self._portfolios.copy()
    
    def get(self, name: str) -> Optional[Portfolio]:
        """
        Get a portfolio by name.
        
        Args:
            name: Portfolio name
            
        Returns:
            Portfolio or None if not found
        """
        self._ensure_loaded()
        return self._portfolios.get(name)
    
    def create(self, portfolio: Portfolio) -> bool:
        """
        Create a new portfolio.
        
        Args:
            portfolio: Portfolio to create
            
        Returns:
            True if successful
        """
        self._ensure_loaded()
        
        if portfolio.name in self._portfolios:
            return False  # Already exists
        
        portfolio.created_at = datetime.now().isoformat()
        portfolio.updated_at = portfolio.created_at
        self._portfolios[portfolio.name] = portfolio
        
        return self.save()
    
    def update(self, portfolio: Portfolio) -> bool:
        """
        Update an existing portfolio.
        
        Args:
            portfolio: Portfolio to update
            
        Returns:
            True if successful
        """
        self._ensure_loaded()
        
        if portfolio.name not in self._portfolios:
            return False  # Not found
        
        portfolio.updated_at = datetime.now().isoformat()
        self._portfolios[portfolio.name] = portfolio
        
        return self.save()
    
    def delete(self, name: str) -> bool:
        """
        Delete a portfolio.
        
        Args:
            name: Portfolio name
            
        Returns:
            True if successful
        """
        self._ensure_loaded()
        
        if name not in self._portfolios:
            return False
        
        del self._portfolios[name]
        return self.save()
    
    def rename(self, old_name: str, new_name: str) -> bool:
        """
        Rename a portfolio.
        
        Args:
            old_name: Current name
            new_name: New name
            
        Returns:
            True if successful
        """
        self._ensure_loaded()
        
        if old_name not in self._portfolios:
            return False
        
        if new_name in self._portfolios:
            return False  # New name already exists
        
        portfolio = self._portfolios.pop(old_name)
        portfolio.name = new_name
        portfolio.updated_at = datetime.now().isoformat()
        self._portfolios[new_name] = portfolio
        
        return self.save()
    
    def import_legacy(self) -> int:
        """
        Import portfolios from legacy portfolios.json file.
        
        Returns:
            Number of portfolios imported
        """
        if not self.legacy_path.exists():
            return 0
        
        self._ensure_loaded()
        imported = 0
        
        try:
            with open(self.legacy_path, 'r', encoding='utf-8') as f:
                legacy_data = json.load(f)
            
            for name, pdata in legacy_data.items():
                # Skip if already exists
                if name in self._portfolios:
                    continue
                
                portfolio = Portfolio.from_legacy_format(name, pdata)
                self._portfolios[name] = portfolio
                imported += 1
            
            if imported > 0:
                self.save()
            
        except Exception as e:
            print(f"Error importing legacy portfolios: {e}")
        
        return imported
    
    def export_to_legacy_format(self, name: str) -> Optional[dict]:
        """
        Export a portfolio to legacy format for compatibility.
        
        Args:
            name: Portfolio name
            
        Returns:
            Dictionary in legacy format or None
        """
        portfolio = self.get(name)
        if portfolio is None:
            return None
        
        return {
            'tickers': portfolio.tickers,
            'weights': portfolio.weights,
        }
    
    def get_portfolio_names(self) -> List[str]:
        """Get list of all portfolio names."""
        self._ensure_loaded()
        return list(self._portfolios.keys())
    
    def duplicate(self, source_name: str, new_name: str) -> bool:
        """
        Duplicate a portfolio with a new name.
        
        Args:
            source_name: Source portfolio name
            new_name: New portfolio name
            
        Returns:
            True if successful
        """
        source = self.get(source_name)
        if source is None:
            return False
        
        if new_name in self._portfolios:
            return False
        
        new_portfolio = Portfolio(
            name=new_name,
            tickers=source.tickers.copy(),
            weights=source.weights.copy(),
            description=f"Copy of {source_name}",
        )
        
        return self.create(new_portfolio)


# Convenience function for quick access
def get_portfolio_manager() -> PortfolioManager:
    """Get a PortfolioManager instance."""
    return PortfolioManager()
