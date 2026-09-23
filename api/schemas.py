from pydantic import BaseModel, Field
from typing import List


class PredictionItem(BaseModel):
    breed: str = Field(
        ...,
        description="Name of the breed"
    )

    confidence: float = Field(
        ...,
        description="Confidence of the model for this breed"
    )


class PredictionResponse(BaseModel):
    predicted_breed: str = Field(
        ...,
        description="Name of the predicted breed"
    )

    confidence: float = Field(
        ...,
        description="Confidence of the model for the predicted breed"
    )

    top_predictions: List[PredictionItem] = Field(
        ...,
        description="Top 3 predicted breeds with their confidence scores"
    )