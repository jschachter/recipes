"""Recipe schema: loose Pydantic models with raw_text fallbacks at every level."""

from pydantic import BaseModel, ConfigDict


class Ingredient(BaseModel):
    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    name: str
    quantity: str | None = None
    unit: str | None = None
    preparation: str | None = None
    notes: str | None = None
    raw_text: str | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int | None = None
    action: str | None = None
    ingredients: list[str] | None = None
    tools: list[str] | None = None
    duration: str | None = None
    temperature: str | None = None
    description: str | None = None
    raw_text: str | None = None


class Recipe(BaseModel):
    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    title: str | None = None
    source: str | None = None
    servings: str | None = None
    total_time: str | None = None
    ingredients: list[Ingredient] | None = None
    steps: list[Step] | None = None
    raw_text: str | None = None


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    completeness: float | None = None
    accuracy: float | None = None
    errors: list[str] | None = None
    commentary: str | None = None
