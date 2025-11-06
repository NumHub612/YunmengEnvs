# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Qualities.
"""
from core.solutions.standards import IQuality, ICategory

from dataclasses import dataclass


@dataclass
class Quality(IQuality):
    """Quality class."""

    def __init__(self, categories: list[ICategory], ordered: bool = False):
        super().__init__()
        self.categories = categories
        self.ordered = ordered

    def has_category(self, category: ICategory) -> bool:
        """Check if the quality has the given category."""
        return category in self.categories

    def count(self) -> int:
        """Count the number of categories."""
        return len(self.categories)

    def get_categories(self, order: int) -> ICategory:
        """Get the category by its order."""
        return self.categories[order]
